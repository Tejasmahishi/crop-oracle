"""
Crop Oracle backend — hybrid web + mobile, built for Karnataka

Run (development):
    uvicorn main:app --reload --port 8000
Run (production): see README.md (Docker or `uvicorn main:app --host 0.0.0.0 --port 8000`).
"""

from __future__ import annotations

import asyncio
import base64
import csv
import io
import importlib
import json
import math
import os
import random
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2lab
from skimage.feature import graycomatrix, graycoprops

import feedback as fb
import guard
import quality


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

APP_VERSION = "3.6-hybrid"

KINDWISE_KEY = (os.getenv("KINDWISE_API_KEY") or os.getenv("KINDWISE_KEY") or "").strip()
KINDWISE_URL = "https://crop.kindwise.com/api/v1/identification"
SARVAM_KEY = (os.getenv("SARVAM_API_KEY") or os.getenv("SARVAM_KEY") or "").strip()

MAX_IMAGE_BYTES = 10 * 1024 * 1024
def _analysis_size(raw: Optional[str]) -> int:
    """ANALYSIS_SIZE from the environment: a whole number of pixels, kept between 160 and 1024. Anything else is 288."""
    try:
        return min(max(int(str(raw).strip()), 160), 1024)
    except (TypeError, ValueError):
        return 288


# The photo is shrunk to this many pixels square before it is measured. Bigger finds smaller spots but is slower, and
# the blur check in quality.py was tuned at 288, so change it only together with scripts/calibrate_quality.py.
IMAGE_SIZE = _analysis_size(os.getenv("ANALYSIS_SIZE"))
MIN_LEAF_FRACTION = 0.03   # below this share of the frame, treat the photo as 'no leaf found'
Image.MAX_IMAGE_PIXELS = 50_000_000

# Where feedback answers and consented photos are kept (mount this folder as a volume).
DATA_DIR = Path(os.getenv("DATA_DIR") or (BASE_DIR / "data"))
try:
    FEEDBACK_RETENTION_DAYS = int(os.getenv("FEEDBACK_RETENTION_DAYS", "730"))   # 0 = keep forever
except ValueError:
    FEEDBACK_RETENTION_DAYS = 730
PRIVACY_CONTACT = (os.getenv("PRIVACY_CONTACT") or "").strip()
PRIVACY_OPERATOR = (os.getenv("PRIVACY_OPERATOR") or "").strip()
# Predictions sent by the browser or phone ("tflite_top3") are ignored unless you opt in,
# because anyone can forge them and so force a "high confidence" result.
ALLOW_CLIENT_ML = os.getenv("ALLOW_CLIENT_ML", "false").strip().lower() in {"1", "true", "yes", "on"}
KVK_PATH = BASE_DIR / "kvk_contacts.json"

SLM_ENABLED = os.getenv("SLM_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
SLM_MODEL = os.getenv("SLM_MODEL", "qwen3:4b").strip()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")

TOP_N_CANDIDATES = 1
MAX_SYMPTOMS_CHARS = 1000   # keep equal to maxlength on the #symptoms textarea in index.html

# Approximate bounding box of Karnataka (with a small margin for the coast).
KA_LAT_MIN, KA_LAT_MAX = 11.5, 18.5
KA_LON_MIN, KA_LON_MAX = 74.0, 78.6


def in_karnataka(lat: float, lon: float) -> bool:
    return KA_LAT_MIN <= lat <= KA_LAT_MAX and KA_LON_MIN <= lon <= KA_LON_MAX

DEBUG_MASKS = os.getenv("DEBUG_MASKS", "false").strip().lower() == "true"


# ============================================================
# OPTIONAL SARVAM  (reserved: translate_text() below is not called anywhere yet;
# all Kannada/Hindi text comes from the built-in templates)
# ============================================================

SARVAM_ON = False
sarvam_client = None
if SARVAM_KEY:
    try:
        from sarvamai import SarvamAI
        sarvam_client = SarvamAI(api_subscription_key=SARVAM_KEY)
        SARVAM_ON = True
    except Exception as e:
        print("[sarvam] unavailable:", repr(e))


# ============================================================
# SHARED HTTP CLIENT
# ============================================================

HTTP_CLIENT: Optional[httpx.AsyncClient] = None

STORE = fb.Store(DATA_DIR / "feedback")
ANALYSIS_SEM: Optional[asyncio.Semaphore] = None   # created at start-up; limits the CPU-heavy part


async def _purge_loop():
    """Delete feedback older than FEEDBACK_RETENTION_DAYS, once a day."""
    while True:
        try:
            removed = await asyncio.to_thread(STORE.purge_older_than, FEEDBACK_RETENTION_DAYS)
            if removed:
                print(f"[feedback] purged {removed} record(s) older than {FEEDBACK_RETENTION_DAYS} days")
        except Exception as e:
            print("[feedback] purge failed:", repr(e))
        await asyncio.sleep(24 * 3600)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global HTTP_CLIENT, ANALYSIS_SEM
    HTTP_CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(25.0, connect=5.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        headers={"User-Agent": f"CropOracle/{APP_VERSION}"},
    )
    ANALYSIS_SEM = asyncio.Semaphore(guard.SETTINGS.max_concurrent_analyses)
    if not PRIVACY_CONTACT:
        print("[privacy] PRIVACY_CONTACT is not set: the /privacy page shows a 'not configured' notice. "
              "Set it before you invite real users.")
    purge_task = asyncio.create_task(_purge_loop())
    try:
        yield
    finally:
        purge_task.cancel()
        await HTTP_CLIENT.aclose()
        HTTP_CLIENT = None
        ANALYSIS_SEM = None


def client() -> httpx.AsyncClient:
    return HTTP_CLIENT or httpx.AsyncClient(timeout=25.0)


# ============================================================
# ASSET LOADING
# ============================================================

def _load_remedies() -> dict:
    path = BASE_DIR / "agronomy_remedies.csv"
    if not path.exists():
        print("[remedies] CSV not found, skipping")
        return {}
    out: dict = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cls = (row.get("Disease") or "").strip()
                if not cls:
                    continue
                entry = {
                    "crop": (row.get("Crop") or "").strip(),
                    "plant_part": (row.get("Plant_Part") or "").strip(),
                    "class": cls,
                    "symptoms": (row.get("Symptoms") or "").strip(),
                    "treatment": (row.get("Verified_Treatment") or "").strip(),
                    "source_org": (row.get("Source_Org") or "").strip(),
                    "source_url": (row.get("Source_URL") or "").strip(),
                }
                out[cls] = entry
                normalized = cls.replace("___", "_").replace(",", "").replace(" ", "_").lower()
                out.setdefault(f"_norm:{normalized}", entry)
    except Exception as e:
        print("[remedies] parse error:", repr(e))
    n = len([k for k in out if not k.startswith("_norm:")])
    print(f"[remedies] loaded {n} entries")
    return out


def _load_tflite_mapping() -> tuple[dict, dict]:
    idx_path = BASE_DIR / "class_indices.json"
    map_path = BASE_DIR / "class_indices_mapping.json"
    idx_to_class: dict = {}
    class_to_meta: dict = {}
    if idx_path.exists():
        try:
            idx_to_class = {int(k): v for k, v in json.loads(idx_path.read_text()).items()}
            print(f"[tflite] loaded {len(idx_to_class)} class indices")
        except Exception as e:
            print("[tflite] class_indices parse error:", repr(e))
    if map_path.exists():
        try:
            raw = json.loads(map_path.read_text())
            class_to_meta = {k: v for k, v in raw.items()
                             if isinstance(v, dict) and not k.startswith("_")}
            n_supported = sum(1 for v in class_to_meta.values() if v.get("id"))
            print(f"[tflite] loaded {len(class_to_meta)} mapping entries · {n_supported} fully supported offline")
        except Exception as e:
            print("[tflite] mapping parse error:", repr(e))
    return idx_to_class, class_to_meta


REMEDIES = _load_remedies()
TFLITE_IDX_TO_CLASS, TFLITE_CLASS_TO_META = _load_tflite_mapping()


# ============================================================
# RICE TFLITE MODEL (trained on srideviklef rice-leaf dataset)
# ============================================================

RICE_MODEL_PATH = BASE_DIR / "rice_model.tflite"
_DEFAULT_RICE_CLASS_NAMES = ["Bacterial leaf blight", "Brown spot", "Health", "Leaf smut"]


def _load_rice_class_names(default: list) -> list:
    """Read the class order from rice_class_indices.json so the names cannot drift from the model.

    Accepts a list, {"0": "name", ...}, or the Keras form {"name": 0, ...}.
    Falls back to the built-in list when the file is missing or unreadable.
    """
    path = BASE_DIR / "rice_class_indices.json"
    if not path.exists():
        return list(default)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            names = [str(x) for x in data]
        elif isinstance(data, dict) and data and all(str(k).isdigit() for k in data):
            names = [str(data[k]) for k in sorted(data, key=int)]
        elif isinstance(data, dict) and data and all(isinstance(v, int) for v in data.values()):
            names = [str(n) for n, _ in sorted(data.items(), key=lambda kv: kv[1])]
        else:
            raise ValueError("unrecognised format")
        print(f"[tflite] rice class names loaded from {path.name}: {names}")
        return names
    except Exception as e:
        print(f"[tflite] {path.name} unreadable ({e!r}); using the built-in class list")
        return list(default)


RICE_CLASS_NAMES = _load_rice_class_names(_DEFAULT_RICE_CLASS_NAMES)
_TFLITE_LOCK = threading.Lock()   # a TFLite interpreter is not safe to call from two threads at once

RICE_INTERPRETER = None
RICE_INPUT_INDEX = None
RICE_OUTPUT_INDEX = None

# How the rice model wants its pixels. The supplied rice_model.tflite is a Keras MobileNetV3Small export, and that
# network rescales 0..255 to -1..1 itself (its first two operations are a multiply and an add), so it must be given
# the raw 0..255 values: "raw", the default. Scaling them here as well squeezed every photo into almost the same
# input, and the model then answered about "68% leaf smut" for any picture. A model exported without its own
# rescaling layer needs RICE_INPUT_SCALE=unit (0..1) or signed (-1..1) in the environment.
_RICE_SCALERS = {
    "raw": lambda a: a,
    "unit": lambda a: a / 255.0,
    "signed": lambda a: (a / 127.5) - 1.0,
}
RICE_INPUT_SCALE = (os.getenv("RICE_INPUT_SCALE") or "raw").strip().lower()
if RICE_INPUT_SCALE not in _RICE_SCALERS:
    print(f"[tflite] RICE_INPUT_SCALE={RICE_INPUT_SCALE!r} is not one of {sorted(_RICE_SCALERS)}; using 'raw'")
    RICE_INPUT_SCALE = "raw"


def _load_rice_model():
    global RICE_INTERPRETER, RICE_INPUT_INDEX, RICE_OUTPUT_INDEX
    if not RICE_MODEL_PATH.exists():
        print(f"[tflite] rice model not found at {RICE_MODEL_PATH}")
        return
    tflite = None
    # tflite-runtime has no wheel for the newest Python versions; ai-edge-litert is its successor.
    for module_name in ("tflite_runtime.interpreter", "ai_edge_litert.interpreter", "tensorflow.lite.python.interpreter"):
        try:
            tflite = importlib.import_module(module_name)
            break
        except ImportError:
            continue
    if tflite is None:
        print("[tflite] none of tflite-runtime, ai-edge-litert or tensorflow is installed, so rice_model.tflite cannot run")
        return
    try:
        RICE_INTERPRETER = tflite.Interpreter(model_path=str(RICE_MODEL_PATH))
        RICE_INTERPRETER.allocate_tensors()
        RICE_INPUT_INDEX = RICE_INTERPRETER.get_input_details()[0]["index"]
        RICE_OUTPUT_INDEX = RICE_INTERPRETER.get_output_details()[0]["index"]
        n_out = int(RICE_INTERPRETER.get_output_details()[0]["shape"][-1])
        if n_out != len(RICE_CLASS_NAMES):
            print(f"[tflite] WARNING: model has {n_out} outputs but {len(RICE_CLASS_NAMES)} class names are "
                  "configured; check rice_class_indices.json")
        print(f"[tflite] rice model loaded ({RICE_MODEL_PATH.name}, "
              f"{RICE_MODEL_PATH.stat().st_size // 1024} KB, input scale: {RICE_INPUT_SCALE})")
        spread = _rice_input_spread()
        if spread is not None and spread < RICE_MIN_PROBE_SPREAD:
            print(f"[tflite] WARNING: the rice model gives almost the same answer for a black, a white, a noise and a "
                  f"green picture (spread {spread:.3f}). That usually means the pixels are scaled wrongly for this "
                  "model. Try RICE_INPUT_SCALE=unit or RICE_INPUT_SCALE=signed (README: 'The rice model').")
    except Exception as e:
        print("[tflite] load failed:", repr(e))
        RICE_INTERPRETER = None


# Four very different pictures must not all get the same answer. The wrong scaling gave a spread of 0.04 on the
# supplied model; the right one gives 0.7. Anything under this is suspicious.
RICE_MIN_PROBE_SPREAD = 0.10


def _rice_input_spread() -> Optional[float]:
    """Largest difference, over the classes, between the model's answers to four very different test pictures.
    Returns None if the model cannot be run."""
    if RICE_INTERPRETER is None:
        return None
    try:
        rng = np.random.default_rng(0)
        probes = [np.zeros((224, 224, 3), np.float32), np.full((224, 224, 3), 255.0, np.float32),
                  rng.uniform(0, 255, (224, 224, 3)).astype(np.float32),
                  np.broadcast_to(np.array([60.0, 135.0, 52.0], np.float32), (224, 224, 3)).copy()]
        outs = []
        with _TFLITE_LOCK:
            for p in probes:
                RICE_INTERPRETER.set_tensor(RICE_INPUT_INDEX, _RICE_SCALERS[RICE_INPUT_SCALE](p)[None])
                RICE_INTERPRETER.invoke()
                outs.append(RICE_INTERPRETER.get_tensor(RICE_OUTPUT_INDEX)[0].astype(np.float64).copy())
        stack = np.stack(outs)
        return float((stack.max(axis=0) - stack.min(axis=0)).max())
    except Exception as e:
        print("[tflite] input check failed:", repr(e))
        return None


def _run_rice_model(image: Image.Image) -> list:
    """Run the trained rice model. Returns top-3 as [{class, confidence}, ...]."""
    if RICE_INTERPRETER is None:
        return []
    try:
        img = image.convert("RGB").resize((224, 224), Image.LANCZOS)
        arr = _RICE_SCALERS[RICE_INPUT_SCALE](np.asarray(img, dtype=np.float32))
        arr = np.expand_dims(arr, axis=0)

        with _TFLITE_LOCK:
            RICE_INTERPRETER.set_tensor(RICE_INPUT_INDEX, arr)
            RICE_INTERPRETER.invoke()
            output = RICE_INTERPRETER.get_tensor(RICE_OUTPUT_INDEX)[0].copy()

        top3 = np.argsort(output)[::-1][:3]
        return [
            {
                "class": RICE_CLASS_NAMES[i] if i < len(RICE_CLASS_NAMES) else f"class_{i}",
                "confidence": float(output[i]),
            }
            for i in top3
        ]
    except Exception as e:
        print("[tflite] inference failed:", repr(e))
        return []


_load_rice_model()
if RICE_INTERPRETER is None:
    print("[tflite] WARNING: no trained model is running. The app is judging by colour, shape and weather only, "
          "and most rice, ragi and arecanut photos will come out with LOW confidence. See README: 'Why is it often low?'")


def find_remedy(crop: str, disease_id: str, tflite_class: Optional[str] = None) -> Optional[dict]:
    if tflite_class and tflite_class in REMEDIES:
        return REMEDIES[tflite_class]
    crop_l = (crop or "").lower()
    did_l = (disease_id or "").lower().replace("_", "")
    for key, entry in REMEDIES.items():
        if key.startswith("_norm:"):
            continue
        if entry["crop"].lower() != crop_l:
            continue
        class_disease = entry["class"].split("___")[-1].lower().replace("_", "").replace(" ", "")
        if did_l and (did_l in class_disease or class_disease in did_l):
            return entry
    return None


# ============================================================
# PLANTVILLAGE INDEX
# ============================================================

class PlantVillageIndex:
    _instance: Optional["PlantVillageIndex"] = None

    def __init__(self, root: Optional[Path] = None):
        self.root = root or (BASE_DIR / "plantvillage" / "raw" / "color")
        self.classes: dict[str, dict] = {}
        self.available = False
        if self.root.exists():
            self._scan()
            self.available = True
        else:
            print(f"[plantvillage] not found at {self.root} — dataset endpoints disabled")

    @classmethod
    def get(cls) -> "PlantVillageIndex":
        if cls._instance is None:
            cls._instance = PlantVillageIndex()
        return cls._instance

    def _scan(self):
        total = 0
        for entry in sorted(os.listdir(self.root)):
            class_dir = self.root / entry
            if not class_dir.is_dir():
                continue
            images = [p for p in os.listdir(class_dir)
                      if p.lower().endswith((".jpg", ".jpeg", ".png"))]
            self.classes[entry] = {
                "count": len(images),
                "sample_paths": [str(class_dir / name) for name in images[:3]],
            }
            total += len(images)
        print(f"[plantvillage] indexed {len(self.classes)} classes · {total} images")

    def count_for(self, class_name: str) -> int:
        return self.classes.get(class_name, {}).get("count", 0)

    def sample_paths(self, class_name: str) -> list[str]:
        return self.classes.get(class_name, {}).get("sample_paths", [])

    def stats(self) -> dict:
        if not self.available:
            return {"available": False}
        counts = {k: v["count"] for k, v in self.classes.items()}
        return {
            "available": True,
            "total_classes": len(self.classes),
            "total_images": sum(counts.values()),
            "min_class_size": min(counts.values()) if counts else 0,
            "max_class_size": max(counts.values()) if counts else 0,
            "per_class": counts,
        }


PLANTVILLAGE = PlantVillageIndex.get()

PV_STATS_PATH = BASE_DIR / "plantvillage_stats.json"
PV_STATS: dict = {}
if PV_STATS_PATH.exists():
    try:
        PV_STATS = json.loads(PV_STATS_PATH.read_text())
        print(f"[plantvillage] loaded stats for {len(PV_STATS)} classes")
    except Exception as e:
        print("[plantvillage] stats load failed:", repr(e))


# ============================================================
# DISEASE DATABASE
# ============================================================

DISEASES = [
    # ---------------- RICE ----------------
    {
        "id": "rice_blast",
        "name": {"en": "Rice Blast", "hi": "धान का ब्लास्ट", "kn": "ಭತ್ತದ ಬ್ಲಾಸ್ಟ್"},
        "pathogen": "Magnaporthe oryzae",
        "crops": ["rice"],
        "pubmed": "Magnaporthe oryzae rice blast India Karnataka",
        "refs": ["uas_bangalore", "icar_rice", "ksda"],
        "envelope": {"temp_opt": 26, "temp_sig": 6, "rh_min": 85, "wet": True},
        "profile": {
            "lesion_ratio":    (0.10,  0.05,  0.15),
            "chlorosis_ratio": (0.03,  0.03,  0.08),
            "brown_ratio":     (0.06,  0.03,  0.12),
            "spot_count":      (8,     6,     0.15),
            "white_ratio":     (0.03,  0.02,  0.15),
            "mean_spot_pct":   (1.8,   0.9,   0.15),
            "mean_aspect":     (3.0,   1.5,   0.15),
            "lab_a_red_frac":  (0.08,  0.05,  0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 2},
            {"t": "fungicide", "active": "tricyclazole", "alt": "carbendazim",
             "dose": "follow product label", "interval": "10"},
            {"t": "airflow"},
        ],
    },
    {
        "id": "rice_blb",
        "name": {"en": "Bacterial Leaf Blight (Rice)",
                 "hi": "धान का जीवाणु झुलसा रोग", "kn": "ಭತ್ತದ ಬ್ಯಾಕ್ಟೀರಿಯಾ ಎಲೆ ಅಂಗಮಾರಿ"},
        "pathogen": "Xanthomonas oryzae pv. oryzae",
        "crops": ["rice"],
        "pubmed": "Xanthomonas oryzae rice bacterial leaf blight India",
        "refs": ["uas_bangalore", "icar_rice", "dppqs"],
        "envelope": {"temp_opt": 28, "temp_sig": 6, "rh_min": 80, "wet": True},
        "profile": {
            "lesion_ratio":    (0.10,  0.06,  0.20),
            "chlorosis_ratio": (0.08,  0.05,  0.15),
            "brown_ratio":     (0.03,  0.02,  0.10),
            "spot_count":      (5,     5,     0.15),
            "white_ratio":     (0.005, 0.005, 0.07),
            "mean_spot_pct":   (1.8,   1.0,   0.20),
            "mean_aspect":     (2.5,   1.5,   0.13),
            "lab_a_red_frac":  (0.15,  0.08,  0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 2},
            {"t": "bactericide", "active": "streptocycline", "alt": "copper",
             "dose": "follow product label", "interval": "10"},
            {"t": "airflow"},
        ],
    },
    {
        "id": "rice_brown_spot",
        "name": {"en": "Rice Brown Spot", "hi": "धान का भूरा धब्बा",
                 "kn": "ಭತ್ತದ ಕಂದು ಚುಕ್ಕೆ"},
        "pathogen": "Bipolaris oryzae",
        "crops": ["rice"],
        "pubmed": "Bipolaris oryzae rice brown spot India",
        "refs": ["uas_bangalore", "icar_rice"],
        "envelope": {"temp_opt": 28, "temp_sig": 7, "rh_min": 80, "wet": True},
        "profile": {
            "lesion_ratio":    (0.04,  0.025, 0.15),
            "chlorosis_ratio": (0.10,  0.05,  0.20),
            "brown_ratio":     (0.04,  0.02,  0.15),
            "spot_count":      (35,    15,    0.20),
            "white_ratio":     (0.002, 0.003, 0.10),
            "mean_spot_pct":   (0.30,  0.20,  0.15),
            "mean_aspect":     (1.2,   0.4,   0.05),
            "lab_a_red_frac":  (0.10,  0.05,  0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 3},
            {"t": "fungicide", "active": "mancozeb", "alt": "carbendazim",
             "dose": "follow product label", "interval": "10"},
            {"t": "soil_test", "days": 14},
        ],
    },
    {
        "id": "rice_leaf_smut",
        "name": {"en": "Rice Leaf Smut", "hi": "धान की पत्ती का कंड रोग",
                 "kn": "ಭತ್ತದ ಎಲೆ ಕಾಡಿಗೆ ರೋಗ"},
        "pathogen": "Entyloma oryzae",
        "crops": ["rice"],
        "pubmed": "Entyloma oryzae rice leaf smut India",
        "refs": ["uas_bangalore", "icar_rice"],
        "envelope": {"temp_opt": 27, "temp_sig": 6, "rh_min": 80, "wet": True},
        "profile": {
            "lesion_ratio":    (0.05,  0.03,  0.15),
            "chlorosis_ratio": (0.05,  0.03,  0.12),
            "brown_ratio":     (0.03,  0.02,  0.15),
            "spot_count":      (25,    12,    0.18),
            "white_ratio":     (0.003, 0.004, 0.05),
            "mean_spot_pct":   (0.35,  0.20,  0.15),
            "mean_aspect":     (1.8,   0.6,   0.20),
            "lab_a_red_frac":  (0.06,  0.04,  0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 2},
            {"t": "fungicide", "active": "carbendazim", "alt": "mancozeb",
             "dose": "follow product label", "interval": "10"},
            {"t": "airflow"},
        ],
    },

    # ---------------- RAGI ----------------
    {
        "id": "ragi_blast",
        "name": {"en": "Ragi Blast (Finger Millet)",
                 "hi": "रागी का ब्लास्ट", "kn": "ರಾಗಿ ಬ್ಲಾಸ್ಟ್"},
        "pathogen": "Pyricularia grisea",
        "crops": ["ragi"],
        "pubmed": "Pyricularia grisea finger millet blast India",
        "refs": ["uas_bangalore", "uas_dharwad", "icar_millets"],
        "envelope": {"temp_opt": 27, "temp_sig": 7, "rh_min": 80, "wet": True},
        "profile": {
            "lesion_ratio":    (0.10,  0.05,  0.15),
            "chlorosis_ratio": (0.03,  0.03,  0.08),
            "brown_ratio":     (0.05,  0.025, 0.12),
            "spot_count":      (8,     6,     0.15),
            "white_ratio":     (0.03,  0.02,  0.15),
            "mean_spot_pct":   (1.6,   0.8,   0.15),
            "mean_aspect":     (2.8,   1.5,   0.15),
            "lab_a_red_frac":  (0.08,  0.05,  0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 2},
            {"t": "fungicide", "active": "carbendazim", "alt": "mancozeb",
             "dose": "follow product label", "interval": "10"},
            {"t": "airflow"},
        ],
    },
    {
        "id": "ragi_leaf_blight",
        "name": {"en": "Ragi Leaf Blight", "hi": "रागी का पत्ती झुलसा रोग",
                 "kn": "ರಾಗಿ ಎಲೆ ಅಂಗಮಾರಿ"},
        "pathogen": "Exserohilum / Helminthosporium spp.",
        "crops": ["ragi"],
        "pubmed": "ragi finger millet leaf blight Karnataka",
        "refs": ["uas_bangalore", "uas_dharwad"],
        "envelope": {"temp_opt": 25, "temp_sig": 7, "rh_min": 75, "wet": True},
        "profile": {
            "lesion_ratio":    (0.12,  0.06,  0.20),
            "chlorosis_ratio": (0.08,  0.05,  0.15),
            "brown_ratio":     (0.08,  0.04,  0.20),
            "spot_count":      (12,    8,     0.15),
            "white_ratio":     (0.005, 0.005, 0.05),
            "mean_spot_pct":   (1.5,   0.8,   0.15),
            "mean_aspect":     (2.2,   1.2,   0.10),
            "lab_a_red_frac":  (0.12,  0.06,  0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 2},
            {"t": "fungicide", "active": "mancozeb", "alt": "carbendazim",
             "dose": "follow product label", "interval": "10"},
            {"t": "airflow"},
        ],
    },

    # ---------------- SUGARCANE ----------------
    {
        "id": "sugarcane_red_rot",
        "name": {"en": "Sugarcane Red Rot", "hi": "गन्ने का लाल सड़न रोग",
                 "kn": "ಕಬ್ಬಿನ ಕೆಂಪು ಕೊಳೆ ರೋಗ"},
        "pathogen": "Colletotrichum falcatum",
        "crops": ["sugarcane"],
        "pubmed": "Colletotrichum falcatum sugarcane red rot India",
        "refs": ["uas_dharwad", "icar_sugarcane"],
        "envelope": {"temp_opt": 30, "temp_sig": 6, "rh_min": 75, "wet": True},
        "profile": {
            "lesion_ratio":    (0.05,  0.04,  0.15),
            "chlorosis_ratio": (0.18,  0.08,  0.35),
            "brown_ratio":     (0.05,  0.03,  0.15),
            "spot_count":      (3,     4,     0.10),
            "white_ratio":     (0.01,  0.01,  0.10),
            "mean_spot_pct":   (0.8,   0.5,   0.10),
            "mean_aspect":     (1.5,   0.7,   0.05),
            "lab_a_red_frac":  (0.15,  0.06,  0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 1},
            {"t": "soil_test", "days": 7},
            {"t": "airflow"},
        ],
    },
    {
        "id": "sugarcane_smut",
        "name": {"en": "Sugarcane Smut", "hi": "गन्ने का कंडुआ रोग",
                 "kn": "ಕಬ್ಬಿನ ಕಾಡಿಗೆ ರೋಗ"},
        "pathogen": "Sporisorium scitamineum",
        "crops": ["sugarcane"],
        "pubmed": "Sporisorium scitamineum sugarcane smut India",
        "refs": ["uas_dharwad", "icar_sugarcane"],
        "envelope": {"temp_opt": 30, "temp_sig": 7, "rh_min": 50, "wet": False},
        "profile": {
            "lesion_ratio":    (0.03,  0.03,  0.15),
            "chlorosis_ratio": (0.06,  0.04,  0.15),
            "brown_ratio":     (0.02,  0.02,  0.10),
            "spot_count":      (2,     3,     0.10),
            "white_ratio":     (0.06,  0.03,  0.35),
            "mean_spot_pct":   (0.4,   0.3,   0.10),
            "mean_aspect":     (1.8,   0.8,   0.05),
            "lab_a_red_frac":  (0.05,  0.04,  0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 1},
            {"t": "soil_test", "days": 7},
            {"t": "airflow"},
        ],
    },

    # ---------------- WHEAT ----------------
    {
        "id": "wheat_rust",
        "name": {"en": "Wheat Rust (Yellow/Leaf/Stem)",
                 "hi": "गेहूँ का रतुआ", "kn": "ಗೋಧಿ ತುಕ್ಕು ರೋಗ"},
        "pathogen": "Puccinia spp.",
        "crops": ["wheat"],
        "pubmed": "Puccinia wheat rust India",
        "refs": ["uas_dharwad", "icar_wheat"],
        "envelope": {"temp_opt": 20, "temp_sig": 6, "rh_min": 70, "wet": True},
        "profile": {
            "lesion_ratio":    (0.10,  0.05,  0.18),
            "chlorosis_ratio": (0.06,  0.04,  0.12),
            "brown_ratio":     (0.04,  0.025, 0.15),
            "spot_count":      (35,    15,    0.20),
            "white_ratio":     (0.005, 0.005, 0.05),
            "mean_spot_pct":   (0.30,  0.20,  0.15),
            "mean_aspect":     (1.8,   0.5,   0.15),
            "lab_a_red_frac":  (0.10,  0.05,  0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 2},
            {"t": "fungicide", "active": "propiconazole", "alt": "hexaconazole",
             "dose": "follow product label", "interval": "15"},
            {"t": "airflow"},
        ],
    },
    {
        "id": "wheat_powdery",
        "name": {"en": "Wheat Powdery Mildew", "hi": "गेहूँ का चूर्णिल आसिता (पाउडरी मिल्ड्यू)",
                 "kn": "ಗೋಧಿ ಬೂದಿ ರೋಗ (ಪೌಡರಿ ಮಿಲ್ಡ್ಯೂ)"},
        "pathogen": "Blumeria graminis f. sp. tritici",
        "crops": ["wheat"],
        "pubmed": "Blumeria graminis wheat powdery mildew India",
        "refs": ["uas_dharwad", "icar_wheat"],
        "envelope": {"temp_opt": 20, "temp_sig": 7, "rh_min": 70, "wet": False},
        "profile": {
            "lesion_ratio":    (0.04,  0.03,  0.10),
            "chlorosis_ratio": (0.04,  0.03,  0.10),
            "brown_ratio":     (0.01,  0.01,  0.05),
            "spot_count":      (8,     8,     0.10),
            "white_ratio":     (0.10,  0.05,  0.50),
            "mean_spot_pct":   (0.6,   0.4,   0.10),
            "mean_aspect":     (1.3,   0.4,   0.05),
            "glcm_homogeneity":(0.72,  0.06,  0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 2},
            {"t": "fungicide", "active": "sulfur", "alt": "hexaconazole",
             "dose": "follow product label", "interval": "10"},
            {"t": "airflow"},
        ],
    },

    # ---------------- COTTON ----------------
    {
        "id": "cotton_leaf_curl",
        "name": {"en": "Cotton Leaf Curl Virus",
                 "hi": "कपास का पत्ती मरोड़ विषाणु रोग",
                 "kn": "ಹತ್ತಿ ಎಲೆ ಮುದುಡು ವೈರಸ್"},
        "pathogen": "Cotton leaf curl virus (CLCuV, whitefly-vectored)",
        "crops": ["cotton"],
        "pubmed": "cotton leaf curl virus India whitefly Karnataka",
        "refs": ["uas_dharwad", "icar_cotton"],
        "envelope": {"temp_opt": 32, "temp_sig": 7, "rh_min": 40, "wet": False},
        "profile": {
            "lesion_ratio":    (0.02,  0.02,  0.10),
            "chlorosis_ratio": (0.14,  0.06,  0.40),
            "brown_ratio":     (0.008, 0.008, 0.10),
            "spot_count":      (4,     5,     0.10),
            "white_ratio":     (0.02,  0.02,  0.15),
            "mean_spot_pct":   (0.3,   0.2,   0.10),
            "mean_aspect":     (1.5,   0.5,   0.05),
            "lab_a_red_frac":  (0.10,  0.05,  0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 1},
            {"t": "fungicide", "active": "imidacloprid", "alt": "neem",
             "dose": "follow product label", "interval": "10"},
            {"t": "airflow"},
        ],
    },
    {
        "id": "cotton_bollworm",
        "name": {"en": "Cotton Bollworm Damage",
                 "hi": "कपास में बॉलवर्म से क्षति",
                 "kn": "ಹತ್ತಿ ಬಾಲ್‌ವರ್ಮ್ ಹಾನಿ"},
        "pathogen": "Helicoverpa armigera / Pectinophora gossypiella",
        "crops": ["cotton"],
        "pubmed": "Helicoverpa armigera cotton India bollworm",
        "refs": ["uas_dharwad", "icar_cotton", "dppqs"],
        "envelope": {"temp_opt": 30, "temp_sig": 8, "rh_min": 40, "wet": False},
        "profile": {
            "lesion_ratio":    (0.08,  0.04,  0.20),
            "chlorosis_ratio": (0.04,  0.03,  0.10),
            "brown_ratio":     (0.08,  0.04,  0.25),
            "spot_count":      (6,     5,     0.15),
            "white_ratio":     (0.005, 0.005, 0.05),
            "mean_spot_pct":   (1.4,   0.7,   0.15),
            "mean_aspect":     (1.4,   0.5,   0.10),
            "glcm_contrast":   (12.0,  5.0,   0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 1},
            {"t": "wash_leaves", "days": 1},
            {"t": "fungicide", "active": "neem", "alt": "imidacloprid",
             "dose": "follow product label", "interval": "7"},
        ],
    },

    # ---------------- GROUNDNUT ----------------
    {
        "id": "groundnut_tikka",
        "name": {"en": "Groundnut Tikka (Leaf Spot)",
                 "hi": "मूँगफली टिक्का रोग", "kn": "ಶೇಂಗಾ ಟಿಕ್ಕಾ ರೋಗ"},
        "pathogen": "Cercospora arachidicola / Cercosporidium personatum",
        "crops": ["groundnut"],
        "pubmed": "Cercospora groundnut tikka leaf spot India Karnataka",
        "refs": ["uas_dharwad", "icar_groundnut"],
        "envelope": {"temp_opt": 28, "temp_sig": 6, "rh_min": 75, "wet": True},
        "profile": {
            "lesion_ratio":    (0.05,  0.03,  0.15),
            "chlorosis_ratio": (0.10,  0.05,  0.20),
            "brown_ratio":     (0.05,  0.03,  0.15),
            "spot_count":      (40,    15,    0.20),
            "white_ratio":     (0.002, 0.003, 0.10),
            "mean_spot_pct":   (0.30,  0.20,  0.15),
            "mean_aspect":     (1.1,   0.3,   0.05),
            "lab_a_red_frac":  (0.10,  0.05,  0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 2},
            {"t": "fungicide", "active": "chlorothalonil", "alt": "mancozeb",
             "dose": "follow product label", "interval": "12"},
            {"t": "airflow"},
        ],
    },
    {
        "id": "groundnut_rust",
        "name": {"en": "Groundnut Rust", "hi": "मूँगफली रतुआ",
                 "kn": "ಶೇಂಗಾ ತುಕ್ಕು"},
        "pathogen": "Puccinia arachidis",
        "crops": ["groundnut"],
        "pubmed": "Puccinia arachidis groundnut rust India",
        "refs": ["uas_dharwad", "icar_groundnut"],
        "envelope": {"temp_opt": 25, "temp_sig": 6, "rh_min": 80, "wet": True},
        "profile": {
            "lesion_ratio":    (0.09,  0.05,  0.18),
            "chlorosis_ratio": (0.06,  0.04,  0.12),
            "brown_ratio":     (0.05,  0.03,  0.15),
            "spot_count":      (28,    15,    0.18),
            "white_ratio":     (0.005, 0.005, 0.07),
            "mean_spot_pct":   (0.35,  0.20,  0.15),
            "mean_aspect":     (1.7,   0.5,   0.15),
            "glcm_contrast":   (14.0,  5.0,   0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 2},
            {"t": "fungicide", "active": "hexaconazole", "alt": "mancozeb",
             "dose": "follow product label", "interval": "12"},
            {"t": "airflow"},
        ],
    },

    # ---------------- CHICKPEA ----------------
    {
        "id": "chickpea_wilt",
        "name": {"en": "Chickpea Wilt (Fusarium)",
                 "hi": "चना उकठा रोग", "kn": "ಕಡಲೆ ಬಾಡುವ ರೋಗ"},
        "pathogen": "Fusarium oxysporum f. sp. ciceris",
        "crops": ["chickpea"],
        "pubmed": "Fusarium oxysporum chickpea wilt India",
        "refs": ["uas_dharwad", "icar_chickpea"],
        "envelope": {"temp_opt": 26, "temp_sig": 6, "rh_min": 40, "wet": False},
        "profile": {
            "lesion_ratio":    (0.02,  0.02,  0.10),
            "chlorosis_ratio": (0.16,  0.07,  0.45),
            "brown_ratio":     (0.02,  0.015, 0.10),
            "spot_count":      (2,     3,     0.10),
            "white_ratio":     (0.005, 0.005, 0.05),
            "mean_spot_pct":   (0.3,   0.2,   0.10),
            "mean_aspect":     (1.3,   0.4,   0.10),
            "lab_a_red_frac":  (0.12,  0.06,  0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 1},
            {"t": "soil_test", "days": 7},
            {"t": "airflow"},
        ],
    },

    # ---------------- ARECANUT ----------------
    {
        "id": "arecanut_mahali",
        "name": {"en": "Arecanut Mahali / Kole Roga",
                 "hi": "सुपारी का महाली रोग", "kn": "ಅಡಿಕೆ ಮಹಾಲಿ ರೋಗ"},
        "pathogen": "Phytophthora arecae (Koleroga)",
        "crops": ["arecanut"],
        "pubmed": "Phytophthora arecae arecanut mahali koleroga India",
        "refs": ["uas_bangalore", "uas_dharwad", "icar_arecanut"],
        "envelope": {"temp_opt": 24, "temp_sig": 6, "rh_min": 85, "wet": True},
        "profile": {
            "lesion_ratio":    (0.12,  0.05,  0.20),
            "chlorosis_ratio": (0.06,  0.04,  0.12),
            "brown_ratio":     (0.10,  0.04,  0.22),
            "spot_count":      (8,     6,     0.15),
            "white_ratio":     (0.03,  0.02,  0.15),
            "mean_spot_pct":   (2.0,   1.0,   0.16),
            "mean_aspect":     (1.4,   0.5,   0.10),
            "glcm_contrast":   (10.0,  4.0,   0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 1},
            {"t": "fungicide", "active": "copper", "alt": "mancozeb",
             "dose": "follow product label", "interval": "15"},
            {"t": "airflow"},
        ],
    },
    {
        "id": "arecanut_yellow_leaf",
        "name": {"en": "Arecanut Yellow Leaf Disease",
                 "hi": "सुपारी का पीली पत्ती रोग", "kn": "ಅಡಿಕೆ ಹಳದಿ ಎಲೆ ರೋಗ"},
        "pathogen": "Phytoplasma (leafhopper-vectored)",
        "crops": ["arecanut"],
        "pubmed": "arecanut yellow leaf disease phytoplasma India",
        "refs": ["uas_bangalore", "icar_arecanut"],
        "envelope": {"temp_opt": 28, "temp_sig": 7, "rh_min": 70, "wet": False},
        "profile": {
            "lesion_ratio":    (0.03,  0.025, 0.10),
            "chlorosis_ratio": (0.22,  0.09,  0.55),
            "brown_ratio":     (0.02,  0.015, 0.10),
            "spot_count":      (3,     4,     0.10),
            "white_ratio":     (0.005, 0.005, 0.05),
            "mean_spot_pct":   (0.4,   0.3,   0.05),
            "mean_aspect":     (1.3,   0.4,   0.05),
            "lab_a_red_frac":  (0.20,  0.08,  0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 1},
            {"t": "fungicide", "active": "imidacloprid", "alt": "neem",
             "dose": "follow product label", "interval": "15"},
            {"t": "soil_test", "days": 14},
        ],
    },
    {
        "id": "arecanut_bud_rot",
        "name": {"en": "Arecanut Bud Rot", "hi": "सुपारी की कली सड़न",
                 "kn": "ಅಡಿಕೆ ಚಿಗುರು ಕೊಳೆ"},
        "pathogen": "Phytophthora palmivora",
        "crops": ["arecanut"],
        "pubmed": "Phytophthora palmivora arecanut bud rot India",
        "refs": ["uas_bangalore", "icar_arecanut"],
        "envelope": {"temp_opt": 25, "temp_sig": 6, "rh_min": 80, "wet": True},
        "profile": {
            "lesion_ratio":    (0.12,  0.05,  0.22),
            "chlorosis_ratio": (0.05,  0.03,  0.10),
            "brown_ratio":     (0.11,  0.05,  0.25),
            "spot_count":      (4,     4,     0.12),
            "white_ratio":     (0.02,  0.015, 0.12),
            "mean_spot_pct":   (2.2,   1.0,   0.14),
            "mean_aspect":     (1.3,   0.4,   0.05),
            "glcm_contrast":   (11.0,  4.0,   0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 1},
            {"t": "fungicide", "active": "copper", "alt": "mancozeb",
             "dose": "follow product label", "interval": "15"},
            {"t": "airflow"},
        ],
    },
    {
        "id": "arecanut_stem_bleeding",
        "name": {"en": "Arecanut Stem Bleeding",
                 "hi": "सुपारी का तना रिसाव रोग", "kn": "ಅಡಿಕೆ ಕಾಂಡ ಸೋರುವಿಕೆ"},
        "pathogen": "Thielaviopsis paradoxa",
        "crops": ["arecanut"],
        "pubmed": "Thielaviopsis paradoxa arecanut stem bleeding India",
        "refs": ["uas_bangalore", "icar_arecanut"],
        "envelope": {"temp_opt": 27, "temp_sig": 6, "rh_min": 70, "wet": True},
        "profile": {
            "lesion_ratio":    (0.06,  0.04,  0.15),
            "chlorosis_ratio": (0.04,  0.03,  0.10),
            "brown_ratio":     (0.08,  0.04,  0.30),
            "spot_count":      (3,     4,     0.10),
            "white_ratio":     (0.005, 0.005, 0.05),
            "mean_spot_pct":   (1.5,   0.8,   0.15),
            "mean_aspect":     (1.6,   0.5,   0.15),
            "glcm_contrast":   (13.0,  5.0,   0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 2},
            {"t": "fungicide", "active": "hexaconazole", "alt": "copper",
             "dose": "follow product label", "interval": "15"},
            {"t": "airflow"},
        ],
    },
    {
        "id": "arecanut_banded_leaf_spot",
        "name": {"en": "Arecanut Banded Leaf Spot",
                 "hi": "सुपारी का पट्टीदार पत्ती धब्बा रोग",
                 "kn": "ಅಡಿಕೆ ಪಟ್ಟಿ ಎಲೆ ಚುಕ್ಕೆ"},
        "pathogen": "Colletotrichum gloeosporioides",
        "crops": ["arecanut"],
        "pubmed": "Colletotrichum arecanut banded leaf spot India",
        "refs": ["uas_bangalore", "icar_arecanut"],
        "envelope": {"temp_opt": 26, "temp_sig": 6, "rh_min": 75, "wet": True},
        "profile": {
            "lesion_ratio":    (0.05,  0.03,  0.15),
            "chlorosis_ratio": (0.06,  0.04,  0.15),
            "brown_ratio":     (0.04,  0.025, 0.15),
            "spot_count":      (18,    10,    0.20),
            "white_ratio":     (0.004, 0.005, 0.05),
            "mean_spot_pct":   (0.35,  0.20,  0.15),
            "mean_aspect":     (2.5,   0.8,   0.15),
            "lab_a_red_frac":  (0.08,  0.05,  0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 2},
            {"t": "fungicide", "active": "mancozeb", "alt": "carbendazim",
             "dose": "follow product label", "interval": "15"},
            {"t": "airflow"},
        ],
    },

    # ---------------- TOMATO / POTATO / PEPPER ----------------
    {
        "id": "early_blight",
        "name": {"en": "Early Blight", "hi": "अगेती झुलसा (अर्ली ब्लाइट)", "kn": "ಅರ್ಲಿ ಬ್ಲೈಟ್"},
        "pathogen": "Alternaria solani",
        "crops": ["tomato", "potato"],
        "pubmed": "tomato early blight Alternaria solani India",
        "refs": ["uas_bangalore", "icar_vegetables"],
        "envelope": {"temp_opt": 26, "temp_sig": 7, "rh_min": 60, "wet": True},
        "profile": {
            "lesion_ratio":    (0.07,  0.04,  0.18),
            "chlorosis_ratio": (0.06,  0.04,  0.15),
            "brown_ratio":     (0.05,  0.03,  0.15),
            "spot_count":      (15,    8,     0.15),
            "white_ratio":     (0.003, 0.005, 0.07),
            "mean_spot_pct":   (0.9,   0.5,   0.15),
            "mean_aspect":     (1.4,   0.5,   0.15),
            "glcm_homogeneity":(0.55,  0.08,  0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 2},
            {"t": "fungicide", "active": "mancozeb", "alt": "chlorothalonil",
             "dose": "follow product label", "interval": "7"},
            {"t": "airflow"},
        ],
    },
    {
        "id": "late_blight",
        "name": {"en": "Late Blight", "hi": "पछेती झुलसा (लेट ब्लाइट)", "kn": "ಲೇಟ್ ಬ್ಲೈಟ್"},
        "pathogen": "Phytophthora infestans",
        "crops": ["tomato", "potato"],
        "pubmed": "Phytophthora infestans tomato late blight India",
        "refs": ["uas_bangalore", "icar_vegetables"],
        "envelope": {"temp_opt": 20, "temp_sig": 5, "rh_min": 80, "wet": True},
        "profile": {
            "lesion_ratio":    (0.15,  0.06,  0.22),
            "chlorosis_ratio": (0.05,  0.04,  0.10),
            "brown_ratio":     (0.10,  0.04,  0.18),
            "spot_count":      (4,     4,     0.10),
            "white_ratio":     (0.02,  0.015, 0.15),
            "mean_spot_pct":   (3.0,   1.2,   0.15),
            "mean_aspect":     (1.8,   0.8,   0.10),
            "glcm_contrast":   (12.0,  5.0,   0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 1},
            {"t": "fungicide", "active": "mancozeb", "alt": "chlorothalonil",
             "dose": "follow product label", "interval": "5"},
            {"t": "airflow"},
        ],
    },
    {
        "id": "bacterial_spot",
        "name": {"en": "Bacterial Spot", "hi": "जीवाणु धब्बा रोग",
                 "kn": "ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಸ್ಪಾಟ್"},
        "pathogen": "Xanthomonas spp.",
        "crops": ["tomato", "pepper"],
        "pubmed": "Xanthomonas bacterial spot tomato chilli India",
        "refs": ["uas_bangalore", "icar_vegetables"],
        "envelope": {"temp_opt": 27, "temp_sig": 7, "rh_min": 70, "wet": True},
        "profile": {
            "lesion_ratio":    (0.05,  0.03,  0.15),
            "chlorosis_ratio": (0.05,  0.03,  0.12),
            "brown_ratio":     (0.045, 0.025, 0.18),
            "spot_count":      (45,    18,    0.25),
            "white_ratio":     (0.001, 0.002, 0.05),
            "mean_spot_pct":   (0.25,  0.15,  0.15),
            "mean_aspect":     (1.3,   0.4,   0.10),
            "lab_a_red_frac":  (0.08,  0.05,  0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 2},
            {"t": "bactericide", "active": "streptocycline", "alt": "copper",
             "dose": "follow product label", "interval": "7"},
            {"t": "airflow"},
        ],
    },
    {
        "id": "powdery_mildew",
        "name": {"en": "Powdery Mildew", "hi": "चूर्णिल आसिता (पाउडरी मिल्ड्यू)",
                 "kn": "ಬೂದಿ ರೋಗ (ಪೌಡರಿ ಮಿಲ್ಡ್ಯೂ)"},
        "pathogen": "Erysiphe / Leveillula spp.",
        "crops": ["tomato", "grape", "pepper"],
        "pubmed": "powdery mildew tomato chilli India",
        "refs": ["uas_bangalore", "icar_vegetables"],
        "envelope": {"temp_opt": 22, "temp_sig": 8, "rh_min": 55, "wet": False},
        "profile": {
            "lesion_ratio":    (0.03,  0.03,  0.10),
            "chlorosis_ratio": (0.03,  0.03,  0.08),
            "brown_ratio":     (0.01,  0.01,  0.05),
            "spot_count":      (10,    8,     0.10),
            "white_ratio":     (0.09,  0.045, 0.42),
            "mean_spot_pct":   (0.6,   0.4,   0.05),
            "mean_aspect":     (1.2,   0.4,   0.05),
            "glcm_homogeneity":(0.75,  0.06,  0.20),
        },
        "actions": [
            {"t": "remove_infected", "days": 2},
            {"t": "fungicide", "active": "sulfur", "alt": "hexaconazole",
             "dose": "follow product label", "interval": "7"},
            {"t": "airflow"},
        ],
    },
    {
        "id": "spider_mite",
        "name": {"en": "Spider Mite Damage", "hi": "मकड़ी (माइट) से क्षति",
                 "kn": "ಸ್ಪೈಡರ್ ಮೈಟ್ ಹಾನಿ"},
        "pathogen": "Tetranychus spp.",
        "crops": ["tomato", "pepper", "cotton"],
        "pubmed": "Tetranychus spider mite India chilli cotton",
        "refs": ["uas_bangalore", "dppqs"],
        "envelope": {"temp_opt": 32, "temp_sig": 8, "rh_min": 25, "wet": False},
        "profile": {
            "lesion_ratio":    (0.03,  0.025, 0.10),
            "chlorosis_ratio": (0.14,  0.05,  0.35),
            "brown_ratio":     (0.01,  0.01,  0.10),
            "spot_count":      (80,    30,    0.20),
            "white_ratio":     (0.02,  0.015, 0.10),
            "mean_spot_pct":   (0.10,  0.08,  0.10),
            "mean_aspect":     (1.2,   0.3,   0.05),
            "lab_a_red_frac":  (0.10,  0.05,  0.10),
        },
        "actions": [
            {"t": "wash_leaves", "days": 1},
            {"t": "fungicide", "active": "neem", "alt": "neem",
             "dose": "follow product label", "interval": "7"},
            {"t": "airflow"},
        ],
    },
    {
        "id": "nitrogen_deficiency",
        "name": {"en": "Nitrogen Deficiency", "hi": "नाइट्रोजन की कमी",
                 "kn": "ಸಾರಜನಕದ ಕೊರತೆ"},
        "pathogen": "Nutrient deficiency",
        "crops": ["tomato", "potato", "pepper", "grape", "rice", "wheat",
                  "cotton", "groundnut", "chickpea", "ragi", "sugarcane", "arecanut"],
        "pubmed": "nitrogen deficiency crop diagnosis India",
        "refs": ["uas_bangalore", "icar_nutrients"],
        "envelope": {"temp_opt": 25, "temp_sig": 10, "rh_min": 0, "wet": False},
        "profile": {
            "lesion_ratio":    (0.005, 0.008, 0.10),
            "chlorosis_ratio": (0.22,  0.08,  0.45),
            "brown_ratio":     (0.003, 0.005, 0.05),
            "spot_count":      (1,     2,     0.10),
            "white_ratio":     (0.003, 0.005, 0.05),
            "mean_spot_pct":   (0.15,  0.10,  0.05),
            "mean_aspect":     (1.0,   0.2,   0.05),
            "lab_a_red_frac":  (0.18,  0.08,  0.20),
        },
        "actions": [
            {"t": "fertilize", "days": 7},
            {"t": "soil_test", "days": 14},
        ],
    },

    # ---------------- GRAPE ----------------
    {
        "id": "grape_downy",
        "name": {"en": "Grape Downy Mildew", "hi": "अंगूर का डाउनी मिल्ड्यू",
                 "kn": "ದ್ರಾಕ್ಷಿಯ ಡೌನಿ ಮಿಲ್ಡ್ಯೂ"},
        "pathogen": "Plasmopara viticola",
        "crops": ["grape"],
        "pubmed": "Plasmopara viticola grape downy mildew India Karnataka",
        "refs": ["uas_bangalore", "icar_grape"],
        "envelope": {"temp_opt": 23, "temp_sig": 6, "rh_min": 75, "wet": True},
        "profile": {
            "lesion_ratio":    (0.06,  0.04,  0.15),
            "chlorosis_ratio": (0.12,  0.05,  0.30),
            "brown_ratio":     (0.02,  0.02,  0.10),
            "spot_count":      (12,    8,     0.15),
            "white_ratio":     (0.04,  0.025, 0.20),
            "mean_spot_pct":   (0.7,   0.4,   0.10),
            "mean_aspect":     (1.3,   0.4,   0.10),
            "lab_a_red_frac":  (0.10,  0.05,  0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 2},
            {"t": "fungicide", "active": "copper", "alt": "mancozeb",
             "dose": "follow product label", "interval": "7"},
            {"t": "airflow"},
        ],
    },
    {
        "id": "grape_black_rot",
        "name": {"en": "Grape Black Rot", "hi": "अंगूर का ब्लैक रॉट",
                 "kn": "ದ್ರಾಕ್ಷಿಯ ಬ್ಲ್ಯಾಕ್ ರಾಟ್"},
        "pathogen": "Guignardia bidwellii",
        "crops": ["grape"],
        "pubmed": "Guignardia bidwellii grape black rot India",
        "refs": ["uas_bangalore", "icar_grape"],
        "envelope": {"temp_opt": 25, "temp_sig": 7, "rh_min": 70, "wet": True},
        "profile": {
            "lesion_ratio":    (0.06,  0.04,  0.18),
            "chlorosis_ratio": (0.03,  0.03,  0.10),
            "brown_ratio":     (0.065, 0.03,  0.25),
            "spot_count":      (12,    8,     0.15),
            "white_ratio":     (0.002, 0.003, 0.07),
            "mean_spot_pct":   (0.7,   0.4,   0.15),
            "mean_aspect":     (1.2,   0.4,   0.10),
            "glcm_contrast":   (11.0,  4.0,   0.10),
        },
        "actions": [
            {"t": "remove_infected", "days": 2},
            {"t": "fungicide", "active": "mancozeb", "alt": "chlorothalonil",
             "dose": "follow product label", "interval": "7"},
            {"t": "airflow"},
        ],
    },
]


DISEASES_BY_CROP: dict[str, list] = {}
DISEASES_BY_ID: dict[str, dict] = {}
for _d in DISEASES:
    DISEASES_BY_ID[_d["id"]] = _d
    _d["_profile_items"] = tuple(_d["profile"].items())
    for _c in _d["crops"]:
        DISEASES_BY_CROP.setdefault(_c, []).append(_d)

SUPPORTED_CROPS: tuple[str, ...] = tuple(sorted(DISEASES_BY_CROP.keys()))
VALID_DISEASE_IDS: frozenset[str] = frozenset(DISEASES_BY_ID.keys())


# ============================================================
# SYMPTOMS
# ============================================================

SYMPTOMS = {
    "rice_blast": ["diamond spots", "spindle spots", "grey centre", "neck rot", "blast", "panicle"],
    "rice_blb": ["yellow leaves", "straw coloured", "wavy margin", "kresek", "bacterial", "wilting"],
    "rice_brown_spot": ["brown spots", "oval spots", "sesame seed", "yellow halo"],
    "rice_leaf_smut": ["black streaks", "smut", "leaf smut", "black spots", "linear lesions"],
    "ragi_blast": ["neck blast", "finger blast", "diamond spots", "grey centre", "blast", "ear head"],
    "ragi_leaf_blight": ["leaf blight", "brown lesions", "elongated spots", "yellowing", "ragi"],
    "sugarcane_red_rot": ["reddening", "internal red", "alcohol smell", "wilting", "drying"],
    "sugarcane_smut": ["whip", "black whip", "smut", "grassy shoot"],
    "wheat_rust": ["yellow stripes", "orange powder", "rust", "yellow rust", "brown rust", "stem rust"],
    "wheat_powdery": ["white powder", "cottony growth", "mildew", "powder"],
    "cotton_leaf_curl": ["curled leaves", "upward curl", "vein thickening", "whitefly", "stunted", "leaf curl"],
    "cotton_bollworm": ["boll damage", "bore holes", "caterpillar", "frass", "rosette", "larva"],
    "groundnut_tikka": ["dark spots", "leaf spot", "tikka", "yellow halo", "defoliation"],
    "groundnut_rust": ["orange pustules", "rust", "powder", "rusty"],
    "chickpea_wilt": ["drooping", "yellowing", "vascular browning", "wilt", "collar rot"],
    "arecanut_mahali": ["kole roga", "mahali", "fruit rot", "bud drop", "water soaked", "rain"],
    "arecanut_yellow_leaf": ["yellow fronds", "yellowing", "leaf yellowing", "stunted", "phytoplasma", "leafhopper"],
    "arecanut_bud_rot": ["bud rot", "rotting bud", "soft rot", "foul smell", "crown rot"],
    "arecanut_stem_bleeding": ["stem bleeding", "dark exudate", "reddish brown", "stem rot", "bleeding"],
    "arecanut_banded_leaf_spot": ["banded spots", "elongated spots", "leaf spot", "yellow halo"],
    "early_blight": ["brown spots", "target spots", "concentric rings", "yellow leaves", "lower leaves"],
    "late_blight": ["water soaked", "dark lesions", "rapid spreading", "white growth", "rain"],
    "bacterial_spot": ["small dark spots", "water soaked", "bacterial", "leaf spots"],
    "powdery_mildew": ["white powder", "white coating", "powder", "mildew"],
    "spider_mite": ["webbing", "fine web", "speckled", "bronzing", "mites"],
    "nitrogen_deficiency": ["yellow leaves", "pale leaves", "yellowing", "slow growth", "nutrient"],
    "grape_downy": ["yellow spots", "white growth", "downy", "rain", "grape"],
    "grape_black_rot": ["brown spots", "black spots", "black rot", "grape"],
}


# ============================================================
# CORPUS
# ============================================================

CORPUS = {
    "uas_bangalore": {"short": "UAS Bangalore", "title": "Crop protection package — UAS Bangalore",
                      "org": "University of Agricultural Sciences, Bengaluru (GKVK)",
                      "url": "https://uasbangalore.edu.in/"},
    "uas_dharwad": {"short": "UAS Dharwad", "title": "Crop protection package — UAS Dharwad",
                    "org": "University of Agricultural Sciences, Dharwad",
                    "url": "https://uasd.edu.in/"},
    "ksda": {"short": "Karnataka Agriculture Dept.", "title": "Karnataka State Department of Agriculture advisories",
             "org": "KSDA — Government of Karnataka",
             "url": "https://raitamitra.karnataka.gov.in/"},
    "kvk_bengaluru": {"short": "KVK Bengaluru Rural", "title": "Krishi Vigyan Kendra — Bengaluru Rural",
                      "org": "ICAR / UAS Bangalore",
                      "url": "https://kvk.icar.gov.in/"},
    "icar_rice": {"short": "ICAR Rice", "title": "Rice disease management — ICAR",
                  "org": "Indian Council of Agricultural Research",
                  "url": "https://icar.org.in/"},
    "icar_millets": {"short": "ICAR-IIMR", "title": "Millets crop protection — ICAR-IIMR",
                     "org": "ICAR — Indian Institute of Millets Research",
                     "url": "https://millets.res.in/"},
    "icar_sugarcane": {"short": "ICAR-IISR", "title": "Sugarcane crop protection — ICAR-IISR",
                       "org": "ICAR — Indian Institute of Sugarcane Research",
                       "url": "https://iisr.icar.gov.in/"},
    "icar_wheat": {"short": "ICAR-IIWBR", "title": "Wheat disease advisory — ICAR-IIWBR",
                   "org": "ICAR — Indian Institute of Wheat & Barley Research",
                   "url": "https://iiwbr.org/"},
    "icar_cotton": {"short": "ICAR-CICR", "title": "Cotton pest and disease management — ICAR-CICR",
                    "org": "ICAR — Central Institute for Cotton Research",
                    "url": "https://cicr.org.in/"},
    "icar_groundnut": {"short": "ICAR-DGR", "title": "Groundnut crop protection — ICAR-DGR",
                       "org": "ICAR — Directorate of Groundnut Research",
                       "url": "https://dgr.org.in/"},
    "icar_chickpea": {"short": "ICAR-IIPR", "title": "Chickpea & pulses protection — ICAR-IIPR",
                      "org": "ICAR — Indian Institute of Pulses Research",
                      "url": "https://iipr.icar.gov.in/"},
    "icar_vegetables": {"short": "ICAR-IIVR", "title": "Vegetable crop protection — ICAR-IIVR",
                        "org": "ICAR — Indian Institute of Vegetable Research",
                        "url": "https://iivr.icar.gov.in/"},
    "icar_grape": {"short": "ICAR-NRCG", "title": "Grape cultivation and protection — ICAR-NRCG",
                   "org": "ICAR — National Research Centre for Grapes, Pune",
                   "url": "https://nrcgrapes.icar.gov.in/"},
    "icar_nutrients": {"short": "ICAR Nutrients", "title": "Soil nutrient management advisories — ICAR",
                       "org": "Indian Council of Agricultural Research",
                       "url": "https://icar.org.in/"},
    "icar_arecanut": {"short": "ICAR-CPCRI", "title": "Arecanut crop protection — ICAR-CPCRI",
                      "org": "ICAR — Central Plantation Crops Research Institute",
                      "url": "https://cpcri.icar.gov.in/"},
    "agropedia": {"short": "Agropedia", "title": "Agropedia — Indian crop advisory portal",
                  "org": "IIT Kanpur / ICAR consortium",
                  "url": "http://agropedia.iitk.ac.in/"},
    "dppqs": {"short": "DPPQS", "title": "Plant protection advisories — DPPQS",
              "org": "Directorate of Plant Protection, Quarantine & Storage",
              "url": "https://ppqs.gov.in/"},
}


# ============================================================
# ACTIONS
# ============================================================

ACTIONS = {
    "fungicide": {
        "icon": "🧪",
        "en": "Spray {active} on the affected plants. Repeat every {interval} days. Follow the label for the dose and the waiting period before harvest.",
        "hi": "{active} का छिड़काव करें। हर {interval} दिन में दोहराएँ। मात्रा और कटाई से पहले की प्रतीक्षा अवधि के लिए लेबल देखें।",
        "kn": "{active} ಸಿಂಪಡಿಸಿ. ಪ್ರತಿ {interval} ದಿನಗಳಿಗೊಮ್ಮೆ ಪುನಃ ಸಿಂಪಡಿಸಿ. ಪ್ರಮಾಣ ಮತ್ತು ಕೊಯ್ಲಿಗೆ ಮೊದಲಿನ ಕಾಯುವ ಅವಧಿಗಾಗಿ ಲೇಬಲ್ ನೋಡಿ.",
    },
    "bactericide": {
        "icon": "🧪",
        "en": "Spray {active} on the affected plants. Repeat every {interval} days. Follow the label for the dose and the waiting period before harvest.",
        "hi": "{active} का छिड़काव करें। हर {interval} दिन में दोहराएँ। मात्रा और कटाई से पहले की प्रतीक्षा अवधि के लिए लेबल देखें।",
        "kn": "{active} ಸಿಂಪಡಿಸಿ. ಪ್ರತಿ {interval} ದಿನಗಳಿಗೊಮ್ಮೆ ಪುನಃ ಸಿಂಪಡಿಸಿ. ಪ್ರಮಾಣ ಮತ್ತು ಕೊಯ್ಲಿಗೆ ಮೊದಲಿನ ಕಾಯುವ ಅವಧಿಗಾಗಿ ಲೇಬಲ್ ನೋಡಿ.",
    },
    "remove_infected": {
        "icon": "✂️",
        "en": "Remove the infected leaves and plant parts within {days} days and bag them. Do not compost them.",
        "hi": "संक्रमित पत्तियाँ और पौधे के हिस्से {days} दिनों के अंदर हटाकर थैली में भर दें। उनसे खाद न बनाएँ।",
        "kn": "ರೋಗಪೀಡಿತ ಎಲೆ ಮತ್ತು ಸಸ್ಯ ಭಾಗಗಳನ್ನು {days} ದಿನಗಳೊಳಗೆ ತೆಗೆದು ಚೀಲದಲ್ಲಿ ಹಾಕಿ. ಅವುಗಳಿಂದ ಕಾಂಪೋಸ್ಟ್ ಮಾಡಬೇಡಿ.",
    },
    "airflow": {
        "icon": "🌬️",
        "en": "Prune the lower leaves and space the plants wider so that air can move through the crop.",
        "hi": "नीचे की पत्तियाँ छाँटें और पौधों के बीच दूरी बढ़ाएँ, ताकि फसल में हवा आ-जा सके।",
        "kn": "ಕೆಳಗಿನ ಎಲೆಗಳನ್ನು ಕತ್ತರಿಸಿ, ಬೆಳೆಯಲ್ಲಿ ಗಾಳಿ ಆಡುವಂತೆ ಸಸ್ಯಗಳ ನಡುವೆ ಹೆಚ್ಚು ಅಂತರ ಬಿಡಿ.",
    },
    "wash_leaves": {
        "icon": "💧",
        "en": "Wash the underside of the leaves with a strong jet of water.",
        "hi": "पत्तियों की निचली सतह को पानी की तेज़ धार से धोएँ।",
        "kn": "ಎಲೆಗಳ ಕೆಳಭಾಗವನ್ನು ಬಲವಾದ ನೀರಿನ ಧಾರೆಯಿಂದ ತೊಳೆಯಿರಿ.",
    },
    "fertilize": {
        "icon": "🌱",
        "en": "Apply a balanced fertiliser. Avoid applying too much nitrogen.",
        "hi": "संतुलित उर्वरक डालें। नाइट्रोजन ज़्यादा न डालें।",
        "kn": "ಸಮತೋಲಿತ ಗೊಬ್ಬರ ಹಾಕಿ. ಸಾರಜನಕವನ್ನು ಹೆಚ್ಚು ಹಾಕಬೇಡಿ.",
    },
    "soil_test": {
        "icon": "🧫",
        "en": "Send a soil sample to your nearest Krishi Vigyan Kendra (KVK) within {days} days.",
        "hi": "{days} दिनों के अंदर मिट्टी का नमूना अपने नज़दीकी कृषि विज्ञान केंद्र (KVK) को भेजें।",
        "kn": "{days} ದಿನಗಳೊಳಗೆ ಮಣ್ಣಿನ ಮಾದರಿಯನ್ನು ನಿಮ್ಮ ಹತ್ತಿರದ ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರಕ್ಕೆ (KVK) ಕಳುಹಿಸಿ.",
    },
    # Added only when the result is uncertain (see apply_input_gate).
    "retake_photos": {
        "icon": "📸",
        "en": "Take two more photos before you decide: one of the underside of the leaf and one of the whole plant. Use daylight and keep the leaf in focus.",
        "hi": "फ़ैसला लेने से पहले दो और फ़ोटो लें: एक पत्ती की निचली सतह की और एक पूरे पौधे की। दिन की रोशनी में लें और पत्ती को फ़ोकस में रखें।",
        "kn": "ನಿರ್ಧರಿಸುವ ಮೊದಲು ಇನ್ನೂ ಎರಡು ಫೋಟೋ ತೆಗೆಯಿರಿ: ಒಂದು ಎಲೆಯ ಕೆಳಭಾಗದ್ದು, ಇನ್ನೊಂದು ಇಡೀ ಸಸ್ಯದ್ದು. ಹಗಲು ಬೆಳಕಿನಲ್ಲಿ, ಎಲೆ ಸ್ಪಷ್ಟವಾಗಿ ಕಾಣುವಂತೆ ತೆಗೆಯಿರಿ.",
    },
    "contact_kvk": {
        "icon": "📞",
        "en": "Show the leaf (or these photos) to your nearest Krishi Vigyan Kendra (KVK) or Raita Samparka Kendra before you spray anything.",
        "hi": "कुछ भी छिड़कने से पहले पत्ती (या ये फ़ोटो) अपने नज़दीकी कृषि विज्ञान केंद्र (KVK) या रैयत संपर्क केंद्र को दिखाएँ।",
        "kn": "ಯಾವುದೇ ಔಷಧ ಸಿಂಪಡಿಸುವ ಮೊದಲು ಎಲೆಯನ್ನು (ಅಥವಾ ಈ ಫೋಟೋಗಳನ್ನು) ನಿಮ್ಮ ಹತ್ತಿರದ ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರ (KVK) ಅಥವಾ ರೈತ ಸಂಪರ್ಕ ಕೇಂದ್ರಕ್ಕೆ ತೋರಿಸಿ.",
    },
}


ACTION_BUCKETS = {
    "remove_infected": "right_now",
    "wash_leaves": "right_now",
    "retake_photos": "right_now",
    "fungicide": "today",
    "bactericide": "today",
    "fertilize": "today",
    "contact_kvk": "today",
    "soil_test": "next_3_days",
    "airflow": "important",
}

# Inputs the farmer would have to buy and apply. They are never suggested when the
# diagnosis is uncertain: a wrong guess costs money and can hurt the crop.
INPUT_ACTIONS = frozenset({"fungicide", "bactericide", "fertilize"})


BUCKET_LABELS = {
    "right_now": {"en": "Right now", "hi": "अभी करें", "kn": "ಈಗಲೇ ಮಾಡಿ"},
    "today": {"en": "Today", "hi": "आज", "kn": "ಇಂದು"},
    "next_3_days": {"en": "Next 3 days", "hi": "अगले 3 दिन", "kn": "ಮುಂದಿನ 3 ದಿನಗಳು"},
    "important": {"en": "Also important", "hi": "यह भी ज़रूरी", "kn": "ಇದು ಕೂಡ ಮುಖ್ಯ"},
}


CONFLICT_TEXT = {
    "ambiguity": {
        "sev": "warning", "icon": "⚠️",
        "en": "The image is not clear-cut: {a} and {b} scored almost the same (score gap: {d} out of 100).",
        "hi": "छवि से साफ़ नतीजा नहीं मिला: {a} और {b} के स्कोर लगभग बराबर हैं (अंतर: 100 में से केवल {d})।",
        "kn": "ಚಿತ್ರದಿಂದ ಸ್ಪಷ್ಟ ಫಲಿತಾಂಶ ಸಿಕ್ಕಿಲ್ಲ: {a} ಮತ್ತು {b} ಅಂಕಗಳು ಬಹುತೇಕ ಸಮ (ವ್ಯತ್ಯಾಸ: 100 ರಲ್ಲಿ ಕೇವಲ {d}).",
    },
    "env_mismatch": {
        "sev": "warning", "icon": "🌦️",
        "en": "The current weather ({t} °C, {rh}% humidity) is a weaker match for {d}.",
        "hi": "मौजूदा मौसम ({t} °C, {rh}% आर्द्रता) {d} के लिए कम अनुकूल है।",
        "kn": "ಪ್ರಸ್ತುತ ಹವಾಮಾನ ({t} °C, {rh}% ಆರ್ದ್ರತೆ) {d} ಗೆ ಅಷ್ಟೊಂದು ಅನುಕೂಲಕರವಲ್ಲ.",
    },
    "rain_mismatch": {
        "sev": "info", "icon": "🌧️",
        "en": "Your symptoms mention rain, but only {r} mm fell in the last 48 hours.",
        "hi": "आपके लक्षणों में बारिश का ज़िक्र है, लेकिन पिछले 48 घंटों में केवल {r} मिमी बारिश हुई।",
        "kn": "ನಿಮ್ಮ ಲಕ್ಷಣಗಳಲ್ಲಿ ಮಳೆಯ ಉಲ್ಲೇಖವಿದೆ, ಆದರೆ ಕಳೆದ 48 ಗಂಟೆಗಳಲ್ಲಿ ಕೇವಲ {r} ಮಿಮೀ ಮಳೆಯಾಗಿದೆ.",
    },
    "reg_sub": {
        "sev": "warning", "icon": "⚖️",
        "en": "{banned} was replaced with {alt} to follow the {region} rules (demo data).",
        "hi": "{region} के नियमों (डेमो डेटा) के अनुसार {banned} की जगह {alt} सुझाया गया है।",
        "kn": "{region} ನಿಯಮಗಳ (ಡೆಮೊ ಮಾಹಿತಿ) ಪ್ರಕಾರ {banned} ಬದಲಿಗೆ {alt} ಸೂಚಿಸಲಾಗಿದೆ.",
    },
    "dry_pathogen": {
        "sev": "info", "icon": "☀️",
        "en": "The weather is wetter than this disease usually prefers.",
        "hi": "मौसम इस रोग के लिए सामान्य से ज़्यादा गीला है।",
        "kn": "ಹವಾಮಾನವು ಈ ರೋಗಕ್ಕೆ ಸಾಮಾನ್ಯವಾಗಿ ಬೇಕಾದುದಕ್ಕಿಂತ ಹೆಚ್ಚು ತೇವವಾಗಿದೆ.",
    },
    "low_signal": {
        "sev": "info", "icon": "🔎",
        "en": "The image shows only weak signs of disease. A clearer close-up of one leaf may improve the result.",
        "hi": "छवि में रोग के संकेत बहुत कमज़ोर दिख रहे हैं। एक पत्ती का साफ़ क्लोज़-अप फ़ोटो नतीजे को बेहतर कर सकता है।",
        "kn": "ಚಿತ್ರದಲ್ಲಿ ರೋಗದ ಚಿಹ್ನೆಗಳು ಬಹಳ ದುರ್ಬಲವಾಗಿವೆ. ಒಂದು ಎಲೆಯ ಸ್ಪಷ್ಟ ಕ್ಲೋಸ್-ಅಪ್ ಫೋಟೋ ಫಲಿತಾಂಶವನ್ನು ಸುಧಾರಿಸಬಹುದು.",
    },
}


T = {
    "en": {
        "healthy": "No active disease detected", "healthy_sub": "Visible signs are below the detection threshold",
        "diagnosis": "Diagnosis", "diagnostic_score": "Diagnosis score", "conf": "Confidence",
        "high": "High", "moderate": "Moderate", "mod": "Moderate", "low": "Low",
        "vision": "Leaf image", "weather": "Weather fit", "symptoms": "Symptom match",
        "no_conflict": "No conflicts found.",
        "evidence": "Evidence", "action_plan": "Action plan", "plan_empty": "No action plan could be generated.",
    },
    "hi": {
        "healthy": "कोई सक्रिय रोग नहीं मिला", "healthy_sub": "दिखने वाले संकेत पहचान-सीमा से नीचे हैं",
        "diagnosis": "रोग की पहचान", "diagnostic_score": "पहचान स्कोर", "conf": "भरोसे का स्तर",
        "high": "ऊँचा", "moderate": "मध्यम", "mod": "मध्यम", "low": "कम",
        "vision": "पत्ती की छवि", "weather": "मौसम से मेल", "symptoms": "लक्षणों से मेल",
        "no_conflict": "कोई विरोधाभास नहीं मिला।",
        "evidence": "साक्ष्य", "action_plan": "कार्य योजना", "plan_empty": "कोई कार्य योजना नहीं बन सकी।",
    },
    "kn": {
        "healthy": "ಯಾವುದೇ ಸಕ್ರಿಯ ರೋಗ ಕಂಡುಬಂದಿಲ್ಲ", "healthy_sub": "ಕಾಣುವ ಚಿಹ್ನೆಗಳು ಪತ್ತೆ ಮಿತಿಗಿಂತ ಕಡಿಮೆ ಇವೆ",
        "diagnosis": "ರೋಗನಿರ್ಣಯ", "diagnostic_score": "ರೋಗನಿರ್ಣಯ ಅಂಕ", "conf": "ವಿಶ್ವಾಸದ ಮಟ್ಟ",
        "high": "ಹೆಚ್ಚು", "moderate": "ಮಧ್ಯಮ", "mod": "ಮಧ್ಯಮ", "low": "ಕಡಿಮೆ",
        "vision": "ಎಲೆ ಚಿತ್ರ", "weather": "ಹವಾಮಾನ ಹೊಂದಾಣಿಕೆ", "symptoms": "ಲಕ್ಷಣ ಹೊಂದಾಣಿಕೆ",
        "no_conflict": "ಯಾವುದೇ ವಿರೋಧಾಭಾಸ ಕಂಡುಬಂದಿಲ್ಲ.",
        "evidence": "ಪುರಾವೆ", "action_plan": "ಕ್ರಿಯಾ ಯೋಜನೆ", "plan_empty": "ಯಾವುದೇ ಕ್ರಿಯಾ ಯೋಜನೆ ರೂಪಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ.",
    },
}


REGION_NAMES = {
    "ka": {"en": "Karnataka", "hi": "कर्नाटक", "kn": "ಕರ್ನಾಟಕ"},
    "eu": {"en": "European Union", "hi": "यूरोपीय संघ", "kn": "ಯುರೋಪಿಯನ್ ಒಕ್ಕೂಟ"},
    "us": {"en": "United States", "hi": "संयुक्त राज्य अमेरिका", "kn": "ಅಮೆರಿಕ ಸಂಯುಕ್ತ ಸಂಸ್ಥಾನ"},
    "br": {"en": "Brazil", "hi": "ब्राज़ील", "kn": "ಬ್ರೆಜಿಲ್"},
    "ke": {"en": "Kenya", "hi": "केन्या", "kn": "ಕೀನ್ಯಾ"},
}


# Used only to label the place returned by reverse geocoding. The rules that
# apply to a result are always the ones the farmer selected (default: Karnataka).
COUNTRY_TO_REGION = {
    "in": "ka", "us": "us", "ca": "us", "br": "br", "ke": "ke",
    "fr": "eu", "de": "eu", "es": "eu", "it": "eu", "nl": "eu",
    "be": "eu", "pt": "eu", "ie": "eu", "at": "eu", "gr": "eu",
}


def _ok_everywhere() -> dict:
    return {"ka": "ok", "eu": "ok", "us": "ok", "br": "ok", "ke": "ok"}


def _restricted_in_eu() -> dict:
    return {"ka": "ok", "eu": "restricted", "us": "ok", "br": "ok", "ke": "ok"}


# Demonstration data only. Karnataka (India) is the home market, so every
# active is "ok" there; the other columns support an export-market check.
REG = {
    "copper": _ok_everywhere(),
    "chlorothalonil": _restricted_in_eu(),
    "sulfur": _ok_everywhere(),
    "neem": _ok_everywhere(),
    "mancozeb": _ok_everywhere(),
    "carbendazim": _restricted_in_eu(),
    "tricyclazole": _ok_everywhere(),
    "hexaconazole": _ok_everywhere(),
    "propiconazole": _ok_everywhere(),
    "streptocycline": _restricted_in_eu(),
    "imidacloprid": _restricted_in_eu(),
}


REG_NOTES = {
    "chlorothalonil": {
        "en": "Chlorothalonil is marked as restricted for this region in the demo rules. Check the current local rules before use.",
        "hi": "डेमो नियमों में इस क्षेत्र के लिए क्लोरोथालोनिल प्रतिबंधित के रूप में दर्ज है। उपयोग से पहले मौजूदा स्थानीय नियम जाँच लें।",
        "kn": "ಡೆಮೊ ನಿಯಮಗಳಲ್ಲಿ ಈ ಪ್ರದೇಶಕ್ಕೆ ಕ್ಲೋರೋಥಾಲೊನಿಲ್ ಅನ್ನು ನಿರ್ಬಂಧಿತ ಎಂದು ಗುರುತಿಸಲಾಗಿದೆ. ಬಳಸುವ ಮೊದಲು ಸ್ಥಳೀಯ ನಿಯಮಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.",
    },
    "carbendazim": {
        "en": "Carbendazim is marked as restricted for this region in the demo rules. Check the current local rules before use.",
        "hi": "डेमो नियमों में इस क्षेत्र के लिए कार्बेन्डाज़िम प्रतिबंधित के रूप में दर्ज है। उपयोग से पहले मौजूदा स्थानीय नियम जाँच लें।",
        "kn": "ಡೆಮೊ ನಿಯಮಗಳಲ್ಲಿ ಈ ಪ್ರದೇಶಕ್ಕೆ ಕಾರ್ಬೆಂಡಾಜಿಮ್ ಅನ್ನು ನಿರ್ಬಂಧಿತ ಎಂದು ಗುರುತಿಸಲಾಗಿದೆ. ಬಳಸುವ ಮೊದಲು ಸ್ಥಳೀಯ ನಿಯಮಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.",
    },
    "streptocycline": {
        "en": "Streptocycline is marked as restricted for this region in the demo rules. Check the current local rules before use.",
        "hi": "डेमो नियमों में इस क्षेत्र के लिए स्ट्रेप्टोसाइक्लिन प्रतिबंधित के रूप में दर्ज है। उपयोग से पहले मौजूदा स्थानीय नियम जाँच लें।",
        "kn": "ಡೆಮೊ ನಿಯಮಗಳಲ್ಲಿ ಈ ಪ್ರದೇಶಕ್ಕೆ ಸ್ಟ್ರೆಪ್ಟೋಸೈಕ್ಲಿನ್ ಅನ್ನು ನಿರ್ಬಂಧಿತ ಎಂದು ಗುರುತಿಸಲಾಗಿದೆ. ಬಳಸುವ ಮೊದಲು ಸ್ಥಳೀಯ ನಿಯಮಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.",
    },
    "imidacloprid": {
        "en": "Imidacloprid is marked as restricted for this region in the demo rules. Check the current local rules before use.",
        "hi": "डेमो नियमों में इस क्षेत्र के लिए इमिडाक्लोप्रिड प्रतिबंधित के रूप में दर्ज है। उपयोग से पहले मौजूदा स्थानीय नियम जाँच लें।",
        "kn": "ಡೆಮೊ ನಿಯಮಗಳಲ್ಲಿ ಈ ಪ್ರದೇಶಕ್ಕೆ ಇಮಿಡಾಕ್ಲೋಪ್ರಿಡ್ ಅನ್ನು ನಿರ್ಬಂಧಿತ ಎಂದು ಗುರುತಿಸಲಾಗಿದೆ. ಬಳಸುವ ಮೊದಲು ಸ್ಥಳೀಯ ನಿಯಮಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.",
    },
}


# English labels are lower-case and carry their own article ("a copper-based
# product") so they read correctly in the middle of a sentence. The renderer
# capitalises the first letter when a sentence starts with one.
ACTIVE_LABEL = {
    "copper": {"en": "a copper-based product", "hi": "कॉपर आधारित उत्पाद", "kn": "ತಾಮ್ರ ಆಧಾರಿತ ಉತ್ಪನ್ನ"},
    "chlorothalonil": {"en": "chlorothalonil", "hi": "क्लोरोथालोनिल", "kn": "ಕ್ಲೋರೋಥಾಲೊನಿಲ್"},
    "sulfur": {"en": "sulphur", "hi": "सल्फर", "kn": "ಸಲ್ಫರ್"},
    "neem": {"en": "a neem-based product", "hi": "नीम आधारित उत्पाद", "kn": "ಬೇವು ಆಧಾರಿತ ಉತ್ಪನ್ನ"},
    "mancozeb": {"en": "mancozeb", "hi": "मैन्कोज़ेब", "kn": "ಮ್ಯಾಂಕೋಜೆಬ್"},
    "carbendazim": {"en": "carbendazim", "hi": "कार्बेन्डाज़िम", "kn": "ಕಾರ್ಬೆಂಡಾಜಿಮ್"},
    "tricyclazole": {"en": "tricyclazole", "hi": "ट्राइसाइक्लाज़ोल", "kn": "ಟ್ರೈಸೈಕ್ಲಾಜೋಲ್"},
    "hexaconazole": {"en": "hexaconazole", "hi": "हेक्साकोनाज़ोल", "kn": "ಹೆಕ್ಸಾಕೊನಾಜೋಲ್"},
    "propiconazole": {"en": "propiconazole", "hi": "प्रोपिकोनाज़ोल", "kn": "ಪ್ರೊಪಿಕೊನಾಜೋಲ್"},
    "streptocycline": {"en": "streptocycline", "hi": "स्ट्रेप्टोसाइक्लिन", "kn": "ಸ್ಟ್ರೆಪ್ಟೋಸೈಕ್ಲಿನ್"},
    "imidacloprid": {"en": "imidacloprid", "hi": "इमिडाक्लोप्रिड", "kn": "ಇಮಿಡಾಕ್ಲೋಪ್ರಿಡ್"},
}


def _active_label(key: str, lang: str) -> str:
    entry = ACTIVE_LABEL.get(key)
    if not entry:
        return key
    return entry.get(lang) or entry["en"]


def _region_label(key: str, lang: str) -> str:
    entry = REGION_NAMES.get(key)
    if not entry:
        return key
    return entry.get(lang) or entry["en"]


# ============================================================
# TRANSLATION
# ============================================================

SARVAM_LANGUAGE = {"en": "en-IN", "hi": "hi-IN", "kn": "kn-IN"}
_ASCII_ONLY = re.compile(r"^[\x00-\x7F\s]*$")


def _strip_markdown(text):
    return re.sub(r"[*_`#]", "", text or "").strip()


def _pick_lang_value(template: dict, lang: str) -> str:
    fallbacks = {"en": ("en",), "hi": ("hi", "en"), "kn": ("kn", "hi", "en")}.get(lang, ("en",))
    for key in fallbacks:
        v = template.get(key)
        if v:
            return v
    return ""


def _needs_translation(text: str) -> bool:
    return bool(text) and bool(_ASCII_ONLY.match(text))


async def translate_text(text, target_lang, source_lang="en"):
    if not text or not SARVAM_ON or target_lang == source_lang:
        return text
    if not _needs_translation(text):
        return text
    target_code = SARVAM_LANGUAGE.get(target_lang)
    if not target_code:
        return text
    try:
        response = await asyncio.to_thread(
            sarvam_client.text.translate,
            input=text,
            source_language_code=SARVAM_LANGUAGE.get(source_lang, "en-IN"),
            target_language_code=target_code,
            model="sarvam-translate:v1",
        )
        translated = getattr(response, "translated_text", None)
        if translated:
            return translated.strip()
    except Exception as e:
        print("[sarvam] translation failed:", repr(e))
    return text


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def _rgb_to_hsv(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc, minc = arr.max(axis=-1), arr.min(axis=-1)
    v = maxc
    delta = maxc - minc
    denom = np.maximum(delta, 1e-6)
    s = np.where(maxc > 0, delta / np.maximum(maxc, 1e-6), 0.0)
    rc, gc, bc = (maxc - r) / denom, (maxc - g) / denom, (maxc - b) / denom
    h = np.where(r == maxc, bc - gc,
        np.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
    h = ((h / 6.0) % 1.0) * 360.0
    return h, s, v


def _resize_padded(image: Image.Image, target: int = IMAGE_SIZE) -> Image.Image:
    w, h = image.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (target, target), (128, 128, 128))
    x0, y0, new_w, new_h = quality.padded_box(w, h, target)
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target, target), (128, 128, 128))
    canvas.paste(resized, (x0, y0))
    return canvas


def _segment_leaf_from_hsv(
    h: np.ndarray, s: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, dict]:
    not_white = ~((s < 0.10) & (v > 0.85))
    not_black = v > 0.22
    foliar_hue = ((h >= 10) & (h <= 185))
    has_sat = s >= 0.12
    mask = not_white & not_black & foliar_hue & has_sat
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5), bool))

    _labeled, _n = ndimage.label(mask)
    if _n > 0:
        _sizes = ndimage.sum(mask, _labeled, range(1, _n + 1))
        _keep = np.where(_sizes >= 200)[0] + 1
        mask = np.isin(_labeled, _keep)

    labeled, n = ndimage.label(mask)
    stats = {"component_count": int(n), "component_sizes": []}
    if n >= 1:
        sizes = np.asarray(ndimage.sum(mask, labeled, range(1, n + 1)), dtype=float)
        order = np.argsort(sizes)[::-1]
        stats["component_sizes"] = [int(sizes[i]) for i in order[:3]]
        if n > 1:
            # Only the largest piece is analysed. The photo-quality check uses the sizes
            # above to warn when several leaves or objects are in the frame.
            stats["largest_component_pct"] = float(sizes[order[0]] / sizes.sum())
            mask = labeled == (int(order[0]) + 1)
    else:
        stats["fallback"] = "no_leaf_found"

    filled = ndimage.binary_fill_holes(mask)
    # Filling holes is meant to close up pale spots inside a leaf. It also swallows the black gaps between
    # overlapping leaves, which were then counted as dead patches. A black area bigger than a spot is a gap.
    gaps = filled & ~mask
    if gaps.any():
        _lab, _n = ndimage.label(gaps)
        _sz = ndimage.sum(gaps, _lab, range(1, _n + 1))
        _big = np.where(_sz > max(40.0, GAP_MIN_SHARE * float(mask.sum())))[0] + 1
        filled &= ~(np.isin(_lab, _big) & (v <= GAP_MAX_V))
    return filled, stats


# A black area inside the leaf outline that covers more than this share of the leaf is a gap, not a spot.
GAP_MIN_SHARE, GAP_MAX_V = 0.015, 0.30
# Dead tissue may reach this far from green tissue, as a share of the frame (see _pale_dead_tissue).
PALE_REACH = 0.15


def _pale_dead_tissue(h: np.ndarray, s: np.ndarray, v: np.ndarray, leaf: np.ndarray) -> np.ndarray:
    """Dried, straw, pink or cream tissue that touches the leaf: tip burn, blight, dead patches.

    The leaf outline only takes green-ish pixels, so this tissue was cut out of the leaf and never counted,
    and a blight lesion is often exactly this colour. A pale area is taken only when every pixel of it lies
    close to green tissue. A sheet of paper or soil reaches far from the leaf, so it is background, not
    damage. Near-white (glare, paper) is never taken. Not checked on real field photos beyond a few."""
    none = np.zeros_like(leaf)
    green = leaf & (h >= 68) & (h <= 175) & (s >= 0.20) & (v > 0.22)
    if not green.any():
        return none
    # Pink and peach (dried leaf) stay fairly pale; straw and cream (hue 35-60) may be a little stronger.
    # Skin sits at hue 15-30 with more colour than that, so a finger on the leaf is mostly left out.
    pinkish = ((h < 35) | (h >= 320)) & (s < 0.30)
    strawish = (h >= 35) & (h < 60) & (s < 0.45)
    cand = (s >= 0.10) & (v >= 0.50) & (pinkish | strawish)
    cand = ndimage.binary_closing(cand, structure=np.ones((3, 3), bool))
    lab, n = ndimage.label(cand)
    if n == 0:
        return none
    ids = np.arange(1, n + 1)
    dist = ndimage.distance_transform_edt(~green)
    far = np.asarray(ndimage.maximum(dist, lab, ids))
    near = np.asarray(ndimage.minimum(dist, lab, ids))
    size = np.asarray(ndimage.sum(cand, lab, ids))
    keep = ids[(far <= PALE_REACH * h.shape[0]) & (near <= 2.5) & (size >= 25)]
    pale = np.isin(lab, keep)
    return pale if pale.sum() <= 0.6 * green.sum() else none


def _drop_edge_fringe(spots: np.ndarray, leaf: np.ndarray, width: int = 3) -> np.ndarray:
    """Remove brownish slivers that sit on the leaf outline (where it meets a dark gap or the soil).

    A piece is dropped only when at least 80% of it lies within `width` pixels of the outline, so a spot
    inside the leaf, however small, is never touched. (An earlier version thinned every spot with a 3x3
    morphological opening, which also erased real spots under about 4 px across.) The edge of the analysis
    frame itself is not treated as an outline."""
    if not spots.any():
        return spots
    inner = ndimage.binary_erosion(leaf, structure=np.ones((3, 3), bool), iterations=width, border_value=1)
    band = leaf & ~inner
    lab, n = ndimage.label(spots, structure=np.ones((3, 3), dtype=int))
    ids = np.arange(1, n + 1)
    total = np.asarray(ndimage.sum(spots, lab, ids))
    in_band = np.asarray(ndimage.sum(spots & band, lab, ids))
    fringe = ids[in_band >= 0.8 * total]
    return spots & ~np.isin(lab, fringe) if len(fringe) else spots


def segment_leaf(rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    return _segment_leaf_from_hsv(*_rgb_to_hsv(rgb))


def _components(mask: np.ndarray, min_size: int = 4):
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return [], []
    slices = ndimage.find_objects(labeled)
    sizes, boxes = [], []
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        size = int((labeled[sl] == (i + 1)).sum())
        if size >= min_size:
            y0, y1 = sl[0].start, sl[0].stop
            x0, x1 = sl[1].start, sl[1].stop
            sizes.append(size)
            boxes.append([int(x0), int(y0), int(x1 - x0), int(y1 - y0), size])
    return sizes, boxes


def _glcm_features(gray_uint8: np.ndarray, mask: np.ndarray) -> dict:
    masked = np.where(mask, gray_uint8, 0).astype(np.uint8)
    quantized = (masked // 16).astype(np.uint8)
    try:
        glcm = graycomatrix(
            quantized,
            distances=[2, 4],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=16, symmetric=True, normed=True,
        )
        return {
            "glcm_contrast": float(graycoprops(glcm, "contrast").mean()),
            "glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
            "glcm_energy": float(graycoprops(glcm, "energy").mean()),
            "glcm_correlation": float(graycoprops(glcm, "correlation").mean()),
        }
    except Exception:
        return {"glcm_contrast": 0.0, "glcm_homogeneity": 0.0,
                "glcm_energy": 0.0, "glcm_correlation": 0.0}


def _decode_image(raw: bytes) -> Image.Image:
    """Validate and decode an uploaded image (CPU-bound; call it from a worker thread)."""
    with Image.open(io.BytesIO(raw)) as probe:
        probe.verify()
    im = Image.open(io.BytesIO(raw))
    if im.format == "JPEG":
        # Lets the JPEG decoder produce a smaller picture straight away, which keeps
        # memory low for very large photos. Never smaller than 2048 px on the long side.
        im.draft("RGB", (2048, 2048))
    return im.convert("RGB")


def analyze_leaf(image: Image.Image, with_extras: bool = False) -> dict:
    """Measure the leaf. With with_extras=True the result also carries "_extras":
    {"quality": photo-quality verdict, "overlay": PNG data URL of the spots found}.
    Remove "_extras" before putting the result in a JSON response."""
    orig_w, orig_h = image.size
    img = _resize_padded(image.convert("RGB"), IMAGE_SIZE)
    rgb = np.asarray(img, dtype=np.float32) / 255.0

    h, s, v = _rgb_to_hsv(rgb)
    leaf_mask, mask_stats = _segment_leaf_from_hsv(h, s, v)
    pale_dead = _pale_dead_tissue(h, s, v, leaf_mask)
    leaf_mask = leaf_mask | pale_dead
    leaf_px = max(int(leaf_mask.sum()), 1)
    inside = leaf_mask

    if inside.any():
        v_leaf = v[inside]
        v_p10 = float(np.percentile(v_leaf, 10))
        dark_threshold = max(0.18, v_p10 * 0.85)
    else:
        dark_threshold = 0.20

    dark = (v < dark_threshold) & inside
    white = (s < 0.16) & (v > 0.55) & inside
    green = (h >= 68) & (h <= 175) & (s >= 0.15) & inside
    yellow = (h >= 38) & (h < 68) & (s >= 0.28) & (v >= 0.42) & inside
    brown = (h >= 0) & (h < 45) & (s >= 0.18) & (v < 0.80) & inside

    # A thin brownish fringe where the leaf meets a dark gap or the background is not a lesion. Only pieces that
    # lie (almost) wholly in that outline band are dropped, so small spots inside the leaf are all kept.
    lesion_mask = _drop_edge_fringe(brown | dark, leaf_mask) | pale_dead

    # Merge nearby fragments
    lesion_mask = ndimage.binary_closing(lesion_mask, structure=np.ones((5, 5), bool))
    _lbl, _n = ndimage.label(lesion_mask)
    if _n > 0:
        _sz = ndimage.sum(lesion_mask, _lbl, range(1, _n + 1))
        _keep = np.where(_sz >= 15)[0] + 1
        lesion_mask = np.isin(_lbl, _keep)

    chlor_mask = yellow
    # Damage = spots, dead patches and yellowing together. Pixels that are only "darker than the rest of the
    # leaf" are left out of this number: a shadow or a dark vein is darker but still green.
    damage_mask = (lesion_mask & ~(dark & ~brown & ~pale_dead)) | chlor_mask
    lesion_sizes, lesion_boxes = _components(lesion_mask, min_size=4)
    chlor_sizes, _ = _components(chlor_mask, min_size=4)
    lesion_boxes.sort(key=lambda b: -b[4])

    aspects, areas = [], []
    for b in lesion_boxes[:20]:
        w = max(b[2], 1)
        h2 = max(b[3], 1)
        aspects.append(max(w, h2) / max(min(w, h2), 1))
        areas.append(b[4])

    mean_spot_px = float(np.mean(lesion_sizes)) if lesion_sizes else 0.0
    mean_aspect = float(np.mean(aspects)) if aspects else 1.0
    p25 = float(np.percentile(areas, 25)) if areas else 0.0
    p50 = float(np.percentile(areas, 50)) if areas else 0.0
    p90 = float(np.percentile(areas, 90)) if areas else 0.0

    lab = rgb2lab(rgb)
    a_channel = lab[..., 1]
    leaf_a = a_channel[inside] if inside.any() else a_channel.ravel()
    a_mean = float(leaf_a.mean())
    a_std = float(leaf_a.std())
    a_red_frac = float((leaf_a > 5).mean())
    a_green_frac = float((leaf_a < -5).mean())

    gray_uint8 = (v * 255).astype(np.uint8)
    glcm = _glcm_features(gray_uint8, inside)

    # Debug save (opt-in)
    if DEBUG_MASKS:
        try:
            from PIL import Image as _PILImage
            debug_dir = BASE_DIR / "debug_masks"
            debug_dir.mkdir(exist_ok=True)
            ts = int(time.time() * 1000)
            _PILImage.fromarray((leaf_mask * 255).astype(np.uint8)).save(
                debug_dir / f"mask_{ts}.png")
            _PILImage.fromarray((lesion_mask * 255).astype(np.uint8)).save(
                debug_dir / f"lesion_{ts}.png")
            _PILImage.fromarray((rgb * 255).astype(np.uint8)).save(
                debug_dir / f"input_{ts}.png")
            print(f"[debug] wrote mask/lesion/input to {debug_dir} (ts={ts})")
        except Exception as e:
            print("[debug] save failed:", repr(e))

    extras = None
    if with_extras:
        pad_box = quality.padded_box(orig_w, orig_h, IMAGE_SIZE)
        extras = {
            "quality": quality.assess(
                orig_w=orig_w, orig_h=orig_h, rgb=rgb, v=v, s=s, leaf_mask=leaf_mask,
                leaf_fraction=leaf_px / (IMAGE_SIZE * IMAGE_SIZE),
                component_sizes=mask_stats.get("component_sizes", []),
                pad_box=pad_box, frame_px=IMAGE_SIZE),
            "overlay": quality.make_overlay_png(lesion_mask, chlor_mask, pad_box),
        }

    result = {
        "leaf_px": leaf_px,
        "leaf_fraction": leaf_px / (IMAGE_SIZE * IMAGE_SIZE),
        "lesion_ratio": float(lesion_mask.sum()) / leaf_px,
        "chlorosis_ratio": float(chlor_mask.sum()) / leaf_px,
        "damage_ratio": float(damage_mask.sum()) / leaf_px,
        "brown_ratio": float(brown.sum()) / leaf_px,
        "white_ratio": float(white.sum()) / leaf_px,
        "green_ratio": float(green.sum()) / leaf_px,
        "spot_count": len(lesion_sizes),
        "mean_spot_pct": mean_spot_px / leaf_px * 100,
        "mean_aspect": mean_aspect,
        "lesion_p25_pct": p25 / leaf_px * 100,
        "lesion_p50_pct": p50 / leaf_px * 100,
        "lesion_p90_pct": p90 / leaf_px * 100,
        "lab_a_mean": a_mean,
        "lab_a_std": a_std,
        "lab_a_red_frac": a_red_frac,
        "lab_a_green_frac": a_green_frac,
        **glcm,
        "_mask_stats": mask_stats,
    }
    if extras is not None:
        result["_extras"] = extras
    return result


# ============================================================
# KINDWISE
# ============================================================

_KINDWISE_MAP = {
    "early blight": "early_blight", "alternaria": "early_blight",
    "late blight": "late_blight", "phytophthora": "late_blight",
    "bacterial spot": "bacterial_spot", "xanthomonas": "bacterial_spot",
    "powdery mildew": "powdery_mildew", "oidium": "powdery_mildew",
    "spider mite": "spider_mite", "tetranychus": "spider_mite",
    "nitrogen deficiency": "nitrogen_deficiency",
    "downy mildew": "grape_downy", "plasmopara": "grape_downy",
    "black rot": "grape_black_rot", "guignardia": "grape_black_rot",
    "rice blast": "rice_blast", "magnaporthe": "rice_blast",
    "leaf blight": "rice_blb", "oryzae": "rice_blb",
    "brown spot": "rice_brown_spot", "bipolaris": "rice_brown_spot",
    "ragi blast": "ragi_blast", "pyricularia": "ragi_blast",
    "wheat rust": "wheat_rust", "puccinia": "wheat_rust",
    "blumeria": "wheat_powdery",
    "leaf curl": "cotton_leaf_curl",
    "bollworm": "cotton_bollworm", "helicoverpa": "cotton_bollworm",
    "tikka": "groundnut_tikka", "cercospora": "groundnut_tikka",
    "groundnut rust": "groundnut_rust", "arachidis": "groundnut_rust",
    "fusarium": "chickpea_wilt",
    "red rot": "sugarcane_red_rot", "colletotrichum": "sugarcane_red_rot",
    "smut": "sugarcane_smut", "sporisorium": "sugarcane_smut",
    "mahali": "arecanut_mahali", "koleroga": "arecanut_mahali",
    "bud rot": "arecanut_bud_rot", "palmivora": "arecanut_bud_rot",
    "stem bleeding": "arecanut_stem_bleeding", "thielaviopsis": "arecanut_stem_bleeding",
}


def _map_kindwise_to_internal(name: str) -> str:
    low = (name or "").lower().strip()
    for key, value in _KINDWISE_MAP.items():
        if key in low:
            return value
    return ""


async def kindwise_analyze(image_bytes: bytes, crop: str, lang: str = "en"):
    if not KINDWISE_KEY or not image_bytes:
        return None
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    try:
        response = await client().post(
            KINDWISE_URL,
            params={"details": "common_names,description,treatment"},
            headers={"Api-Key": KINDWISE_KEY, "Content-Type": "application/json"},
            json={"images": [image_b64]},
        )
        if response.status_code != 200:
            print("[kindwise] HTTP:", response.status_code, response.text[:200])
            return None
        data = response.json()
        suggestions = ((data.get("result") or {}).get("disease") or {}).get("suggestions") or []
        top = next((s for s in suggestions if isinstance(s, dict) and s.get("name")), None)
        if not top:
            return None
        disease_name = top.get("name") or ""
        disease_id = _map_kindwise_to_internal(disease_name)
        raw_conf = top.get("probability", top.get("confidence", 0.0))
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        if disease_id not in VALID_DISEASE_IDS:
            disease_id = ""
        return {"source": "kindwise", "disease_id": disease_id,
                "disease_name": disease_name, "confidence": confidence, "crop": crop}
    except Exception as e:
        print("[kindwise] failed:", repr(e))
        return None


# ============================================================
# WEATHER
# ============================================================

WEATHER_FALLBACK = {
    "temp": 26.0, "rh": 72.0, "precip_now": 0.0, "wind": 8.0, "code": 3,
    "rain_48h": 8.4, "rain_7d": 22.0, "rh_mean": 70.0,
    "t_max": 30.0, "t_min": 20.0, "rain_next_48h": 3.2, "rain_prob": 45.0,
    "tz": "Asia/Kolkata", "live": False,
}


async def fetch_weather(lat: float, lon: float):
    params = {
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,precipitation_probability_max",
        "past_days": 7, "forecast_days": 3, "timezone": "Asia/Kolkata",
    }
    try:
        response = await client().get("https://api.open-meteo.com/v1/forecast", params=params)
        response.raise_for_status()
        raw = response.json()
        current = raw.get("current") or {}
        daily = raw.get("daily") or {}
        times = daily.get("time") or []
        if not times:
            raise ValueError("No weather dates.")
        today = (current.get("time") or "")[:10]
        try:
            idx = times.index(today)
        except ValueError:
            idx = min(7, len(times) - 1)
        precip = daily.get("precipitation_sum") or []
        humidity_mean = daily.get("relative_humidity_2m_mean") or []
        temp_max = daily.get("temperature_2m_max") or []
        temp_min = daily.get("temperature_2m_min") or []
        rain_prob = daily.get("precipitation_probability_max") or []

        def num(arr, pos, default=0.0):
            try:
                v = arr[pos]
                return default if v is None else float(v)
            except (IndexError, TypeError, ValueError):
                return default

        rain_48h = sum(num(precip, i) for i in range(max(0, idx - 2), idx))
        rain_7d = sum(num(precip, i) for i in range(max(0, idx - 7), idx))
        rain_next_48h = sum(num(precip, i) for i in range(idx + 1, min(len(precip), idx + 3)))

        def cur(key, fallback):
            try:
                v = current.get(key)
                return float(fallback) if v is None else float(v)
            except (TypeError, ValueError):
                return float(fallback)

        try:
            weather_code = int(current.get("weather_code", WEATHER_FALLBACK["code"]))
        except (TypeError, ValueError):
            weather_code = WEATHER_FALLBACK["code"]

        return {
            "temp": cur("temperature_2m", WEATHER_FALLBACK["temp"]),
            "rh": cur("relative_humidity_2m", WEATHER_FALLBACK["rh"]),
            "precip_now": cur("precipitation", 0),
            "wind": cur("wind_speed_10m", WEATHER_FALLBACK["wind"]),
            "code": weather_code,
            "rain_48h": rain_48h, "rain_7d": rain_7d,
            "rh_mean": num(humidity_mean, idx, WEATHER_FALLBACK["rh_mean"]),
            "t_max": num(temp_max, idx, WEATHER_FALLBACK["t_max"]),
            "t_min": num(temp_min, idx, WEATHER_FALLBACK["t_min"]),
            "rain_next_48h": rain_next_48h,
            "rain_prob": num(rain_prob, idx + 1, 0),
            "tz": raw.get("timezone", "Asia/Kolkata"),
            "live": True,
        }
    except Exception as e:
        print("[weather] failed:", repr(e))
        return dict(WEATHER_FALLBACK)


# Results are cached briefly by ~1 km cell, and identical requests that arrive together share one
# upstream call. Only successful (live) results are cached, so a temporary outage is never remembered.
WEATHER_TTL_SEC = 600
GEOCODE_TTL_SEC = 3600
_CACHE_MAX = 512
_weather_cache: dict = {}
_geocode_cache: dict = {}
_inflight: dict = {}


async def _cached_once(cache: dict, key, ttl: float, make_call, cacheable):
    now = time.time()
    hit = cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    fut = _inflight.get((id(cache), key))
    if fut is None:
        fut = asyncio.ensure_future(make_call())
        _inflight[(id(cache), key)] = fut

        def _finished(f, key=key):
            _inflight.pop((id(cache), key), None)
            if f.cancelled() or f.exception() is not None:
                return
            value = f.result()
            if cacheable(value):
                if len(cache) >= _CACHE_MAX:
                    cache.pop(min(cache, key=lambda k: cache[k][0]), None)
                cache[key] = (time.time(), value)
        fut.add_done_callback(_finished)
    return await asyncio.shield(fut)


async def fetch_weather_cached(lat: float, lon: float):
    result = await _cached_once(_weather_cache, (round(lat, 2), round(lon, 2)), WEATHER_TTL_SEC,
                                lambda: fetch_weather(lat, lon), lambda r: bool(r.get("live")))
    return dict(result)


async def reverse_geocode_cached(lat: float, lon: float):
    result = await _cached_once(_geocode_cache, (round(lat, 2), round(lon, 2)), GEOCODE_TTL_SEC,
                                lambda: reverse_geocode(lat, lon), lambda r: r is not None)
    return dict(result) if result else None


def weather_suitability(env: dict, wx: dict) -> float:
    try:
        t, rh, rain = float(wx["temp"]), float(wx["rh"]), float(wx["rain_48h"])
        temp_score = math.exp(-((t - env["temp_opt"]) ** 2) / (2 * env["temp_sig"] ** 2))
        rh_score = 1.0 if env["rh_min"] <= 0 else max(0.0, min(1.0, (rh - (env["rh_min"] - 25)) / 25))
        if env["wet"]:
            wet_score = min(1.0, 0.25 + rain / 12 + (0.25 if rh > 85 else 0))
        else:
            wet_score = max(0.0, min(1.0, 0.85 - rain / 25 - (0.35 if rh > 88 else 0)))
        return max(0.0, min(1.0, 0.5 * temp_score + 0.3 * rh_score + 0.2 * wet_score))
    except Exception:
        return 0.0


# ============================================================
# GEOCODING
# ============================================================

async def reverse_geocode(lat: float, lon: float):
    try:
        response = await client().get(
            "https://api.bigdatacloud.net/data/reverse-geocode-client",
            params={"latitude": lat, "longitude": lon, "localityLanguage": "en"},
            timeout=7.0,
        )
        response.raise_for_status()
        data = response.json()
        label = ", ".join(filter(None, [
            data.get("locality") or data.get("city"),
            data.get("principalSubdivision"),
            data.get("countryName"),
        ])) or f"{lat:.3f}, {lon:.3f}"
        cc = (data.get("countryCode") or "").lower()
        return {"label": label, "country_code": cc, "region": COUNTRY_TO_REGION.get(cc)}
    except Exception as e:
        print("[geocode] failed:", repr(e))
        return None


# ============================================================
# PUBMED
# ============================================================

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


async def pubmed_search(term: str, retmax: int = 4):
    if not term:
        return []
    try:
        c = client()
        search = await c.get(
            _EUTILS + "esearch.fcgi",
            params={"db": "pubmed", "retmode": "json", "retmax": retmax,
                    "sort": "relevance", "term": term},
            timeout=8.0,
        )
        search.raise_for_status()
        ids = search.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        summary = await c.get(
            _EUTILS + "esummary.fcgi",
            params={"db": "pubmed", "retmode": "json", "id": ",".join(ids)},
            timeout=8.0,
        )
        summary.raise_for_status()
        result = summary.json().get("result", {})
        output = []
        for pmid in ids:
            record = result.get(pmid)
            if not record:
                continue
            authors = ", ".join(a.get("name", "") for a in record.get("authors", [])[:3]
                                if isinstance(a, dict))
            year = (record.get("pubdate", "") or "")[:4]
            journal = record.get("fulljournalname") or record.get("source") or "PubMed"
            output.append({
                "kind": "pubmed", "pmid": pmid,
                "title": record.get("title", "(untitled)"),
                "org": f"{journal} - {year}" if year else journal,
                "authors": authors,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "score": 0.90,
            })
        return output
    except Exception as e:
        print("[pubmed] failed:", repr(e))
        return []


def corpus_chunks(ref_ids):
    output = []
    for index, ref_id in enumerate(ref_ids):
        entry = CORPUS.get(ref_id)
        if not entry:
            continue
        output.append({
            "kind": "corpus", "id": ref_id,
            "short": entry.get("short", ""), "title": entry["title"], "org": entry["org"], "url": entry["url"],
            "score": max(0.40, min(0.95, 0.94 - index * 0.045)),
        })
    return output


PUBMED_TTL_SEC = 6 * 3600
_pubmed_cache: dict = {}   # PubMed gets the same 28 searches over and over; remember good answers


async def retrieve(disease, live=True):
    citations, live_count, failed = [], 0, False
    if live:
        term = disease["pubmed"]
        pubmed_results = await _cached_once(
            _pubmed_cache, term, PUBMED_TTL_SEC,
            lambda: pubmed_search(term, 4), lambda r: bool(r))
        if pubmed_results:
            citations.extend(pubmed_results)
            live_count = len(pubmed_results)
        else:
            failed = True
    citations.extend(corpus_chunks(disease["refs"]))
    citations.sort(key=lambda c: -c["score"])
    return citations, live_count, failed


# ============================================================
# REGULATION
# ============================================================

def check_regulation(disease, region, lang="en"):
    if region not in REGION_NAMES:
        region = "ka"
    steps, substitutions, flags = [], [], []
    seen = set()
    for action in disease["actions"]:
        action_type = action["t"]
        if action_type in ("fungicide", "bactericide"):
            primary = action["active"]
            alternative = action.get("alt")
            status = REG.get(primary, {}).get(region, "unknown")
            chosen, substituted = primary, False
            if status != "ok" and alternative and REG.get(alternative, {}).get(region) == "ok":
                chosen, substituted = alternative, True
            if chosen not in seen:
                seen.add(chosen)
                steps.append({
                    "type": "spray", "action": action_type, "active": chosen,
                    "dose": action.get("dose", "follow label"),
                    "interval": action.get("interval"),
                    "substituted": substituted,
                })
            if substituted:
                substitutions.append({"banned": primary, "alt": chosen})
                note = REG_NOTES.get(primary, {}).get(lang) or REG_NOTES.get(primary, {}).get("en")
                flags.append(note or f"{primary} is not approved in this region.")
        else:
            if action_type not in seen:
                seen.add(action_type)
                steps.append({"type": "cultural", "action": action})
    return {
        "steps": steps, "substitutions": substitutions, "flags": flags,
        "region_code": region, "region_name": _region_label(region, lang),
    }


def apply_input_gate(reg: dict, withheld: bool) -> dict:
    """When the diagnosis is uncertain, drop every spray and fertiliser step.

    What is left are the harmless steps (remove infected leaves, improve airflow,
    soil test), with two steps added in front: take better photos and show the
    leaf to a KVK. Pesticide-substitution notes are dropped too, because they
    describe products that are no longer being suggested.
    """
    if not withheld:
        return reg
    steps = [{"type": "cultural", "action": {"t": "retake_photos"}},
             {"type": "cultural", "action": {"t": "contact_kvk"}}]
    for step in reg["steps"]:
        if step["type"] == "spray":
            continue
        if step["type"] == "cultural" and step["action"]["t"] in INPUT_ACTIONS:
            continue
        steps.append(step)
    return {**reg, "steps": steps, "substitutions": [], "flags": [], "spray_withheld": True}


# ============================================================
# SCORING
# ============================================================

def _gauss(x: float, mu: float, sig: float) -> float:
    sig = max(float(sig), 1e-6)
    return math.exp(-((x - mu) ** 2) / (2 * sig * sig))


def _score_profile(feat: dict, profile_items: tuple) -> float:
    num = den = 0.0
    for key, (mu, sig, weight) in profile_items:
        value = feat.get(key, 0.0)
        num += weight * _gauss(value, mu, sig)
        den += weight
    return num / den if den else 0.0


_SYMPTOM_LOWER = {did: tuple(k.lower() for k in kws) for did, kws in SYMPTOMS.items()}


# Farmers often describe symptoms in Kannada or Hindi. The disease keywords above are in
# English, so common Kannada/Hindi words are mapped to concepts and the concepts to the
# English keyword phrases. This is a starter vocabulary: extend the tuples as needed.
import unicodedata as _ud


def _nfc(text: str) -> str:
    return _ud.normalize("NFC", text or "").replace("\u200c", "").replace("\u200d", "")


_SYMPTOM_CONCEPTS = {
    "brown":  ("ಕಂದು", "भूर"),
    "yellow": ("ಹಳದಿ", "पील", "पीला", "पीली", "पीले"),
    "white":  ("ಬಿಳಿ", "ಬಿಳುಪು", "सफेद", "सफ़ेद"),
    "black":  ("ಕಪ್ಪು", "काला", "काले", "काली"),
    "spot":   ("ಚುಕ್ಕೆ", "ಮಚ್ಚೆ", "ಕಲೆ", "धब्बे", "धब्बा", "दाग"),
    "powder": ("ಪುಡಿ", "पाउडर", "चूर्ण"),
    "rust":   ("ತುಕ್ಕು", "रतुआ", "जंग", "ज़ंग"),
    "wilt":   ("ಬಾಡ", "मुरझा"),
    "dry":    ("ಒಣಗ", "सूख"),
    "curl":   ("ಸುರುಳಿ", "ಮುದುಡ", "ಮುದುರ", "मुड़", "सिकुड़"),
    "rot":    ("ಕೊಳೆ", "सड़"),
    "bud":    ("ಸುಳಿ", "ಮೊಗ್ಗು", "कली"),
    "stem":   ("ಕಾಂಡ", "तना"),
    "fruit":  ("ಕಾಯಿ", "फल"),
    "web":    ("ಜೇಡರ", "जाला"),
    "rain":   ("ಮಳೆ", "बारिश", "वर्षा"),
    "stunt":  ("ಕುಂಠಿತ", "बौन", "विकास रुक"),
    "fly":    ("ನೊಣ", "मक्खी"),
    "worm":   ("ಹುಳ", "इल्ली", "सूंडी"),
    "blast":  ("ಬೆಂಕಿ ರೋಗ", "ಬ್ಲಾಸ್ಟ್", "ब्लास्ट", "झोंका"),
    "kole":   ("ಕೊಳೆ ರೋಗ", "ಮಹಾಳಿ", "ಮಹಾಲಿ"),
}
_SYMPTOM_CONCEPTS = {c: tuple(_nfc(w) for w in ws) for c, ws in _SYMPTOM_CONCEPTS.items()}

_SYMPTOM_RULES = [
    (("brown", "spot"),   "brown spots brown lesions"),
    (("black", "spot"),   "black spots dark spots"),
    (("yellow", "spot"),  "yellow spots"),
    (("spot",),           "leaf spot leaf spots"),
    (("yellow",),         "yellow leaves yellowing"),
    (("white", "powder"), "white powder"),
    (("powder",),         "powder"),
    (("rust",),           "rust rusty"),
    (("wilt",),           "wilting wilt drooping"),
    (("dry",),            "drying"),
    (("curl",),           "leaf curl"),
    (("rot", "bud"),      "bud rot rotting bud"),
    (("rot", "stem"),     "stem rot"),
    (("rot", "fruit"),    "fruit rot"),
    (("rot",),            "soft rot"),
    (("kole",),           "kole roga"),
    (("web",),            "webbing fine web"),
    (("rain",),           "rain"),
    (("stunt",),          "stunted slow growth"),
    (("white", "fly"),    "whitefly"),
    (("worm",),           "caterpillar larva"),
    (("blast",),          "blast"),
]


def _normalise_symptoms(text: str) -> str:
    """Lower-case the text and append English keywords for any Kannada/Hindi terms in it."""
    low = _nfc(text).lower()
    if not low or low.isascii():
        return low
    found = {c for c, words in _SYMPTOM_CONCEPTS.items() if any(w in low for w in words)}
    extra = [phrase for needs, phrase in _SYMPTOM_RULES if all(n in found for n in needs)]
    return low + (" " + " ".join(extra) if extra else "")


def _symptom_match(disease: dict, text: str) -> float:
    low = _normalise_symptoms(text)
    if not low:
        return 0.35
    hits = sum(1 for kw in _SYMPTOM_LOWER.get(disease["id"], ()) if kw in low)
    return max(0.0, min(1.0, 0.35 + hits * 0.20))


def _plantvillage_class_match(features: dict, class_name: str) -> float:
    stats = PV_STATS.get(class_name)
    if not stats:
        return 1.0
    keys = ("lesion_ratio", "chlorosis_ratio", "brown_ratio",
            "white_ratio", "mean_spot_pct", "mean_aspect")
    diffs = []
    for k in keys:
        want = stats.get(k, 0.0)
        got = features.get(k, 0.0)
        scale = max(want, 0.05) if k != "mean_aspect" else max(want, 1.0)
        diffs.append(abs(got - want) / scale)
    mean_diff = sum(diffs) / len(diffs)
    return max(0.0, 1.0 - min(mean_diff, 2.0) / 2.0)


def tflite_hint_to_internal(top3: list[dict]) -> tuple[Optional[str], float, list[dict]]:
    if not top3:
        return None, 0.0, []
    evidence = []
    best_id, best_conf = None, 0.0
    for entry in top3[:3]:
        cls = (entry.get("class") or "").strip()
        try:
            conf = float(entry.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        meta = TFLITE_CLASS_TO_META.get(cls)
        internal = (meta or {}).get("id")
        evidence.append({
            "class": cls,
            "confidence": round(conf, 3),
            "internal_id": internal,
        })
        if internal and conf > best_conf:
            best_id, best_conf = internal, conf
    return best_id, best_conf, evidence


def rank_candidates(
    feat: dict, crop: str, wx: dict, symptoms: str,
    tflite_id: Optional[str], tflite_conf: float, tflite_class_name: Optional[str],
    kindwise_id: Optional[str], kindwise_conf: float,
) -> list[dict]:
    crop = (crop or "rice").lower().strip()
    pool = DISEASES_BY_CROP.get(crop, [])
    output = []
    for d in pool:
        vision = _score_profile(feat, d["_profile_items"])
        weather = weather_suitability(d["envelope"], wx)
        symptom = _symptom_match(d, symptoms)
        ml_boost = 0.0
        if tflite_id and tflite_id == d["id"]:
            pv_match = _plantvillage_class_match(feat, tflite_class_name or "")
            calibrated = tflite_conf * (0.5 + 0.5 * pv_match)
            ml_boost = max(ml_boost, calibrated * 0.85)
        if kindwise_id and kindwise_id == d["id"]:
            ml_boost = max(ml_boost, kindwise_conf * 1.0)
        base = 0.50 * vision + 0.15 * weather + 0.15 * symptom + 0.20 * ml_boost
        output.append({**d, "vision_score": vision, "weather_suit": weather,
                       "symptom_score": symptom, "ml_boost": ml_boost, "score": base})
    output.sort(key=lambda x: -x["score"])
    return output


# ============================================================
# CONFLICTS
# ============================================================

def find_conflicts(top, second, wx, symptoms_text, reg, feat, region_code, lang="en"):
    conflicts = []
    nm = lambda d: d["name"].get(lang) or d["name"]["en"]
    if second and (top["score"] - second["score"]) < 0.10 and top["score"] > 0.25:
        conflicts.append({"key": "ambiguity", "vars": {
            "a": nm(top), "b": nm(second),
            "d": abs(round((top["score"] - second["score"]) * 100))}})
    if top["weather_suit"] < 0.50 and top["vision_score"] > 0.58:
        conflicts.append({"key": "env_mismatch", "vars": {
            "d": nm(top), "t": f'{wx["temp"]:.1f}', "rh": str(round(wx["rh"]))}})
    symptom_lower = (symptoms_text or "").lower()
    if (any(w in symptom_lower for w in ["rain", "rained", "storm", "बारिश", "ಮಳೆ"])
            and wx["rain_48h"] < 1.5):
        conflicts.append({"key": "rain_mismatch", "vars": {"r": f'{wx["rain_48h"]:.1f}'}})
    if reg.get("substitutions"):
        sub = reg["substitutions"][0]
        conflicts.append({"key": "reg_sub", "vars": {
            "banned": _active_label(sub["banned"], lang),
            "alt": _active_label(sub["alt"], lang),
            "region": _region_label(region_code, lang)}})
    if top["envelope"]["wet"] is False and wx["rain_48h"] > 6:
        conflicts.append({"key": "dry_pathogen", "vars": {}})
    if (feat.get("spot_count", 0) == 0
            and feat.get("lesion_ratio", 0) < 0.012
            and top["vision_score"] < 0.55):
        conflicts.append({"key": "low_signal", "vars": {}})
    return conflicts


def render_conflicts(conflicts, lang):
    output = []
    for conflict in conflicts:
        spec = CONFLICT_TEXT.get(conflict["key"])
        if not spec:
            continue
        text = _pick_lang_value(spec, lang)
        for key, value in conflict["vars"].items():
            text = text.replace("{" + key + "}", str(value))
        text = text[:1].upper() + text[1:]
        output.append({"key": conflict["key"], "sev": spec["sev"],
                       "icon": spec["icon"], "text": text})
    return output


# ============================================================
# PLAN RENDERING
# ============================================================

def _render_action(action, lang="en"):
    if lang not in ("en", "hi", "kn"):
        lang = "en"
    if action["type"] == "spray":
        template = ACTIONS.get(action["action"], ACTIONS["fungicide"])
        text = _pick_lang_value(template, lang)
        text = text.replace("{active}", _active_label(action["active"], lang))
        text = text.replace("{dose}", str(action.get("dose", "follow label")))
        text = text.replace("{interval}", str(action.get("interval", "")))
        return {"icon": template["icon"], "text": text, "key": action["action"]}
    data = action["action"]
    template = ACTIONS.get(data["t"])
    if not template:
        return {"icon": "•", "text": "", "key": data["t"]}
    text = _pick_lang_value(template, lang).replace("{days}", str(data.get("days", 7)))
    return {"icon": template["icon"], "text": text, "key": data["t"]}


def build_plan(top, reg, disease_refs, corpus, lang="en"):
    buckets = {"right_now": [], "today": [], "next_3_days": [], "important": []}
    for step in reg["steps"]:
        key = step["action"]["t"] if step["type"] == "cultural" else step["action"]
        bucket = ACTION_BUCKETS.get(key)
        if not bucket:
            continue
        rendered = _render_action(step, lang)
        if not rendered["text"]:
            continue
        refs = []
        # "Take more photos" and "ask a KVK" are not disease advice, so they carry no source links.
        if key not in ("retake_photos", "contact_kvk"):
            for ref_id in disease_refs[:2]:
                entry = corpus.get(ref_id)
                if entry:
                    refs.append({"short": entry.get("short", ""), "title": entry["title"], "org": entry["org"], "url": entry["url"]})
        rendered["refs"] = refs
        buckets[bucket].append(rendered)
    return buckets


# ============================================================
# SPOKEN
# ============================================================

_SPOKEN_HEALTHY = {
    "en": "Your crop looks healthy. Check it again in five to seven days.",
    "hi": "आपकी फसल स्वस्थ दिख रही है। पाँच से सात दिन बाद दोबारा जाँच करें।",
    "kn": "ನಿಮ್ಮ ಬೆಳೆ ಆರೋಗ್ಯವಾಗಿ ಕಾಣುತ್ತಿದೆ. ಐದರಿಂದ ಏಳು ದಿನಗಳ ನಂತರ ಮತ್ತೊಮ್ಮೆ ಪರಿಶೀಲಿಸಿ.",
}


_SPOKEN_UNCERTAIN = {
    "en": "We are not sure about this result. Our best guess is {name}, but please do not spray anything yet.",
    "hi": "इस नतीजे के बारे में हम पक्के नहीं हैं। हमारा सबसे अच्छा अनुमान {name} है, लेकिन अभी कुछ भी न छिड़कें।",
    "kn": "ಈ ಫಲಿತಾಂಶದ ಬಗ್ಗೆ ನಮಗೆ ಖಚಿತವಿಲ್ಲ. ನಮ್ಮ ಉತ್ತಮ ಊಹೆ {name}, ಆದರೆ ಈಗಲೇ ಏನನ್ನೂ ಸಿಂಪಡಿಸಬೇಡಿ.",
}
_SPOKEN_INCONCLUSIVE = {
    "en": "We could not check this photo reliably. Please take a clearer photo in daylight, with one leaf filling the frame, or show the leaf to your nearest Krishi Vigyan Kendra.",
    "hi": "हम इस फ़ोटो को भरोसे से जाँच नहीं सके। कृपया दिन की रोशनी में, एक पत्ती से पूरा फ़्रेम भरकर साफ़ फ़ोटो लें, या पत्ती अपने नज़दीकी कृषि विज्ञान केंद्र को दिखाएँ।",
    "kn": "ಈ ಫೋಟೋವನ್ನು ವಿಶ್ವಾಸಾರ್ಹವಾಗಿ ಪರಿಶೀಲಿಸಲು ಆಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಹಗಲು ಬೆಳಕಿನಲ್ಲಿ, ಒಂದು ಎಲೆ ಚೌಕಟ್ಟು ತುಂಬುವಂತೆ ಸ್ಪಷ್ಟ ಫೋಟೋ ತೆಗೆಯಿರಿ, ಅಥವಾ ಎಲೆಯನ್ನು ನಿಮ್ಮ ಹತ್ತಿರದ ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರಕ್ಕೆ ತೋರಿಸಿ.",
}


def build_spoken(top, is_healthy, plan_buckets, lang, disease_name_local=None,
                 uncertain=False, inconclusive=False):
    strip = lambda text: re.sub(r"<[^>]+>", "", text or "").replace("**", "")
    if lang not in ("en", "hi", "kn"):
        lang = "en"
    if inconclusive:
        return _SPOKEN_INCONCLUSIVE[lang]
    if is_healthy:
        return _SPOKEN_HEALTHY[lang]
    score = round(top["score"] * 100)
    name = disease_name_local or top["name"]["en"]
    if uncertain:
        parts = [_SPOKEN_UNCERTAIN[lang].format(name=name)]
    else:
        parts = [{
            "en": f"Diagnosis: {name}. Diagnosis score {score} percent.",
            "hi": f"रोग की पहचान: {name}। पहचान स्कोर {score} प्रतिशत।",
            "kn": f"ರೋಗನಿರ್ಣಯ: {name}. ರೋಗನಿರ್ಣಯ ಅಂಕ ಶೇಕಡಾ {score}.",
        }[lang]]
    number = 1
    for bucket in ("right_now", "today", "next_3_days", "important"):
        for step in plan_buckets.get(bucket, []):
            parts.append(f"{number}. {strip(step['text'])}")
            number += 1
    return " ".join(parts)


# ============================================================
# SLM AGENT
# ============================================================

SLM_LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "kn": "Kannada"}

_FACTOR_LABELS = {
    "leaf": {"en": "Leaf image", "hi": "पत्ती की छवि", "kn": "ಎಲೆ ಚಿತ್ರ"},
    "weather": {"en": "Weather", "hi": "मौसम", "kn": "ಹವಾಮಾನ"},
    "symptoms": {"en": "Your symptoms", "hi": "आपके बताए लक्षण", "kn": "ನೀವು ತಿಳಿಸಿದ ಲಕ್ಷಣಗಳು"},
    "second": {"en": "Other possibility", "hi": "दूसरी संभावना", "kn": "ಇನ್ನೊಂದು ಸಾಧ್ಯತೆ"},
}
_FALLBACK_HEADLINE_HEALTHY = {
    "en": "No strong sign of disease was found.",
    "hi": "रोग का कोई ठोस संकेत नहीं मिला।",
    "kn": "ರೋಗದ ಯಾವುದೇ ಸ್ಪಷ್ಟ ಚಿಹ್ನೆ ಕಂಡುಬಂದಿಲ್ಲ.",
}
_FALLBACK_EXPLAIN_HEALTHY = {
    "en": "The leaf shows no clear signs of disease. Keep watching your crop and check it again in 5–7 days.",
    "hi": "पत्ती में रोग के कोई साफ़ संकेत नहीं दिख रहे। फसल पर नज़र रखें और 5–7 दिन बाद दोबारा जाँच करें।",
    "kn": "ಎಲೆಯಲ್ಲಿ ರೋಗದ ಯಾವುದೇ ಸ್ಪಷ್ಟ ಚಿಹ್ನೆ ಕಾಣುತ್ತಿಲ್ಲ. ಬೆಳೆಯನ್ನು ಗಮನಿಸುತ್ತಿರಿ ಮತ್ತು 5–7 ದಿನಗಳ ನಂತರ ಮತ್ತೊಮ್ಮೆ ಪರಿಶೀಲಿಸಿ.",
}
_FALLBACK_HEADLINE = {
    "en": "{d} is the most likely diagnosis.",
    "hi": "{d} सबसे संभावित रोग है।",
    "kn": "{d} ಅತ್ಯಂತ ಸಂಭಾವ್ಯ ರೋಗ.",
}
# Explanation = lead + (symptom part, only when the farmer typed symptoms) + tail.
_FALLBACK_EXPLAIN = {
    "en": {
        "lead": "This result combines the leaf image ({v}% match) and weather suitability ({w}%)",
        "sym": " and your symptom description ({s}%)",
        "tail": ". Please confirm it in the field before spraying.",
    },
    "hi": {
        "lead": "यह नतीजा पत्ती की छवि ({v}% मेल) और मौसम की अनुकूलता ({w}%)",
        "sym": " तथा आपके बताए लक्षणों ({s}%)",
        "tail": " पर आधारित है। छिड़काव से पहले खेत में इसकी पुष्टि कर लें।",
    },
    "kn": {
        "lead": "ಈ ಫಲಿತಾಂಶವು ಎಲೆ ಚಿತ್ರ ({v}% ಹೊಂದಾಣಿಕೆ) ಮತ್ತು ಹವಾಮಾನ ಅನುಕೂಲತೆ ({w}%)",
        "sym": " ಹಾಗೂ ನೀವು ತಿಳಿಸಿದ ಲಕ್ಷಣಗಳ ({s}%)",
        "tail": " ಆಧಾರದ ಮೇಲಿದೆ. ಸಿಂಪಡಿಸುವ ಮೊದಲು ಹೊಲದಲ್ಲಿ ಇದನ್ನು ದೃಢಪಡಿಸಿಕೊಳ್ಳಿ.",
    },
}
_FALLBACK_LEAF_DETAIL = {
    "en": "{n} spot clusters · {x}% of the leaf is spotted.",
    "hi": "{n} धब्बों के समूह · पत्ती का {x}% हिस्सा धब्बेदार है।",
    "kn": "{n} ಚುಕ್ಕೆ ಗುಂಪುಗಳು · ಎಲೆಯ {x}% ಭಾಗದಲ್ಲಿ ಚುಕ್ಕೆಗಳಿವೆ.",
}
_FALLBACK_WEATHER_DETAIL = {
    "en": "{t} °C · {rh}% humidity · {r} mm rain in 48 h.",
    "hi": "{t} °C · {rh}% आर्द्रता · 48 घंटे में {r} मिमी बारिश।",
    "kn": "{t} °C · {rh}% ಆರ್ದ್ರತೆ · 48 ಗಂಟೆಗಳಲ್ಲಿ {r} ಮಿಮೀ ಮಳೆ.",
}
_ESTIMATED_SUFFIX = {
    "en": " (estimated: live weather was unavailable)",
    "hi": " (अनुमानित: लाइव मौसम उपलब्ध नहीं था)",
    "kn": " (ಅಂದಾಜು: ನೇರ ಹವಾಮಾನ ಮಾಹಿತಿ ಲಭ್ಯವಿರಲಿಲ್ಲ)",
}
_FALLBACK_CAUTION = {
    "en": ("This is decision support, not a laboratory test. For severe or uncertain cases, "
           "confirm with your nearest Krishi Vigyan Kendra or Raita Samparka Kendra."),
    "hi": ("यह निर्णय में सहायता है, प्रयोगशाला जाँच नहीं। गंभीर या अनिश्चित मामलों में अपने नज़दीकी "
           "कृषि विज्ञान केंद्र या रैयत संपर्क केंद्र से पुष्टि करें।"),
    "kn": ("ಇದು ನಿರ್ಧಾರಕ್ಕೆ ನೆರವು ನೀಡುವ ಸಾಧನ, ಪ್ರಯೋಗಾಲಯ ಪರೀಕ್ಷೆಯಲ್ಲ. ತೀವ್ರ ಅಥವಾ ಅನಿಶ್ಚಿತ ಪ್ರಕರಣಗಳಲ್ಲಿ "
           "ನಿಮ್ಮ ಹತ್ತಿರದ ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರ ಅಥವಾ ರೈತ ಸಂಪರ್ಕ ಕೇಂದ್ರದಲ್ಲಿ ದೃಢಪಡಿಸಿಕೊಳ್ಳಿ."),
}


def _fallback_reasoning(top, second, features, weather, symptoms_text, is_healthy, language):
    lang = language if language in ("en", "hi", "kn") else "en"
    caution = _FALLBACK_CAUTION[lang]
    if is_healthy:
        return {"headline": _FALLBACK_HEADLINE_HEALTHY[lang],
                "explanation": _FALLBACK_EXPLAIN_HEALTHY[lang],
                "factors": [], "caution": caution}
    dname = top["name"].get(lang) or top["name"]["en"]
    has_symptoms = bool(symptoms_text.strip())
    weather_detail = _FALLBACK_WEATHER_DETAIL[lang].format(
        t=f"{weather['temp']:.1f}", rh=f"{weather['rh']:.0f}", r=f"{weather['rain_48h']:.1f}")
    if not weather.get("live", True):
        weather_detail += _ESTIMATED_SUFFIX[lang]
    factors = [
        {"label": _FACTOR_LABELS["leaf"][lang],
         "detail": _FALLBACK_LEAF_DETAIL[lang].format(
             n=features["spot_count"], x=f"{features['lesion_ratio'] * 100:.1f}"),
         "strength": round(top["vision_score"], 2)},
        {"label": _FACTOR_LABELS["weather"][lang],
         "detail": weather_detail,
         "strength": round(top["weather_suit"], 2)},
    ]
    if has_symptoms:
        factors.append({"label": _FACTOR_LABELS["symptoms"][lang],
                        "detail": symptoms_text.strip()[:180],
                        "strength": round(top["symptom_score"], 2)})
    parts = _FALLBACK_EXPLAIN[lang]
    explanation = parts["lead"].format(
        v=round(top["vision_score"] * 100), w=round(top["weather_suit"] * 100))
    if has_symptoms:
        explanation += parts["sym"].format(s=round(top["symptom_score"] * 100))
    explanation += parts["tail"]
    return {"headline": _FALLBACK_HEADLINE[lang].format(d=dname),
            "explanation": explanation, "factors": factors, "caution": caution}


async def run_slm_agent(top, second, features, weather, symptoms_text, is_healthy, language):
    fallback = _fallback_reasoning(top, second, features, weather, symptoms_text, is_healthy, language)
    if not SLM_ENABLED:
        return {"enabled": False, "used": False, "model": SLM_MODEL,
                "mode": "evidence_fallback", "reasoning": fallback}
    evidence = {
        "diagnosis": top["name"]["en"], "pathogen": top["pathogen"],
        "diagnostic_score": round(top["score"], 3),
        "vision_score": round(top["vision_score"], 3),
        "weather_score": round(top["weather_suit"], 3),
        "symptom_score": round(top["symptom_score"], 3),
        "leaf": {"spot_count": features["spot_count"],
                 "lesion_ratio": round(features["lesion_ratio"], 4),
                 "chlorosis_ratio": round(features["chlorosis_ratio"], 4)},
        "weather": {"temp_c": round(weather["temp"], 1),
                    "humidity_pct": round(weather["rh"], 1),
                    "rain_48h_mm": round(weather["rain_48h"], 1)},
        "farmer_symptoms": symptoms_text.strip()[:500],
        "second_candidate": second["name"]["en"] if second else None,
        "healthy": is_healthy, "region_context": "Karnataka, India",
    }
    prompt = f"""You are Crop Oracle's Decision Agent for Karnataka farmers. Explain structured evidence; do not invent facts.
Language: {SLM_LANGUAGE_NAMES.get(language, 'English')}.
Return ONLY valid JSON with keys: headline, explanation, factors, caution.
factors must be a JSON array of objects with label, detail, strength (0 to 1).
Keep the explanation farmer-friendly and under 90 words. Never prescribe a pesticide dose.
Write ALL output in the specified language. Use Indian crop and disease terminology.
Evidence:\n{json.dumps(evidence, ensure_ascii=False)}"""
    payload = {
        "model": SLM_MODEL, "stream": False, "format": "json",
        "messages": [
            {"role": "system", "content": "You are an evidence-grounded agricultural decision explainer for Karnataka farmers."},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0.1, "num_predict": 320},
    }
    try:
        r = await client().post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=8.0)
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "")
        parsed = json.loads(content)
        if not isinstance(parsed, dict) or not parsed.get("explanation"):
            raise ValueError("SLM returned incomplete JSON")
        parsed.setdefault("factors", [])
        parsed.setdefault("caution", fallback.get("caution"))
        return {"enabled": True, "used": True, "model": SLM_MODEL,
                "mode": "ollama", "reasoning": parsed}
    except Exception as e:
        print("[slm] unavailable:", repr(e))
        return {"enabled": True, "used": False, "model": SLM_MODEL,
                "mode": "evidence_fallback", "error": str(e), "reasoning": fallback}


HEALTH_LABELS = {
    "healthy": {"en": "Looks healthy", "hi": "स्वस्थ दिख रही है", "kn": "ಆರೋಗ್ಯವಾಗಿ ಕಾಣುತ್ತಿದೆ"},
    "good": {"en": "Low concern", "hi": "कम चिंता", "kn": "ಕಡಿಮೆ ಕಾಳಜಿ"},
    "warn": {"en": "Moderate concern", "hi": "मध्यम चिंता", "kn": "ಮಧ್ಯಮ ಕಾಳಜಿ"},
    "bad": {"en": "High concern", "hi": "ज़्यादा चिंता", "kn": "ಹೆಚ್ಚಿನ ಕಾಳಜಿ"},
}


def build_health_score(top, weather, features, is_healthy, lang="en"):
    """How much of the leaf in the photo is damaged: spots, dead patches and yellowing, as a percentage.

    This replaces an older 0-100 "health" number that was a re-mix of the diagnosis and weather scores (the
    weather counted twice). It comes from the picture alone, so the weather and the diagnosis do not move it.
    It is a rough estimate: shadows are ignored, but a very pale or black patch, a finger or soil may still be
    counted or missed. The bands (under 5 % low, under 25 % moderate) are rules of thumb, not calibrated.
    `top` and `weather` are unused and kept so existing callers still work."""
    pick = lambda key: HEALTH_LABELS[key].get(lang) or HEALTH_LABELS[key]["en"]
    pct = int(round(100 * max(0.0, min(1.0, float(features.get("damage_ratio", 0.0))))))
    if is_healthy:
        level, label = "good", pick("healthy")
    else:
        level = "bad" if pct >= 25 else "warn" if pct >= 5 else "good"
        label = pick(level)
    return {"score": pct, "label": label, "level": level, "type": "leaf_damage_percent", "is_demo": True}


# ============================================================
# PRECOMPUTE
# ============================================================

def precompute_class_statistics(out_path: Optional[Path] = None):
    out_path = out_path or PV_STATS_PATH
    if not PLANTVILLAGE.available:
        print("[plantvillage] dataset not installed, skipping precompute")
        return
    out: dict[str, dict] = {}
    for class_name in PLANTVILLAGE.classes:
        class_dir = PLANTVILLAGE.root / class_name
        files = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg"))]
        random.seed(42)
        sample = random.sample(files, min(50, len(files)))
        accum: dict[str, float] = {}
        n = 0
        for fname in sample:
            try:
                with Image.open(class_dir / fname) as im:
                    feats = analyze_leaf(im.convert("RGB"))
                for k in ("lesion_ratio", "chlorosis_ratio", "brown_ratio",
                          "white_ratio", "mean_spot_pct", "mean_aspect",
                          "lab_a_red_frac"):
                    accum[k] = accum.get(k, 0.0) + feats.get(k, 0.0)
                n += 1
            except Exception:
                continue
        if n == 0:
            continue
        out[class_name] = {k: v / n for k, v in accum.items()}
        out[class_name]["_samples"] = n
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[plantvillage] wrote stats for {len(out)} classes to {out_path}")


# ============================================================
# FASTAPI APP
# ============================================================

_DOCS_ON = os.getenv("ENABLE_DOCS", "false").strip().lower() in {"1", "true", "yes", "on"}
app = FastAPI(
    title="Crop Oracle", version=APP_VERSION, lifespan=lifespan,
    docs_url="/docs" if _DOCS_ON else None,
    redoc_url=None,
    openapi_url="/openapi.json" if _DOCS_ON else None,
)


@app.middleware("http")
async def guard_middleware(request: Request, call_next):
    """Rate limit and size-check API calls before the body is read; add security headers."""
    path = request.url.path
    if path.startswith("/api/"):
        ip = guard.client_ip(request.headers, request.client.host if request.client else None,
                             guard.SETTINGS.trusted_proxies)
        bucket, limit = guard.bucket_for(request.method, path)
        allowed, wait = guard.LIMITER.check(bucket, ip, limit)
        if not allowed:
            return JSONResponse({"detail": "Too many requests. Please wait a moment and try again."},
                                status_code=429, headers={"Retry-After": str(wait)})
        if request.method in ("POST", "PUT", "PATCH"):
            declared = request.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > guard.SETTINGS.max_body_bytes:
                return JSONResponse({"detail": "Request too large (the photo can be at most 10 MB)."},
                                    status_code=413)
    response = await call_next(request)
    for name, value in guard.security_headers(response.headers.get("content-type", "")).items():
        response.headers.setdefault(name, value)
    if path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


def configure_cors(target: FastAPI, origins: list) -> None:
    """Allow other sites to call the API only if you list them in CORS_ALLOWED_ORIGINS.

    The web page is served by this same app, so by default no cross-site access is allowed.
    Added after guard_middleware, so it wraps it and error replies carry the CORS headers too.
    """
    if not origins:
        return
    target.add_middleware(
        CORSMiddleware,
        allow_origins=origins, allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Delete-Token", "X-Admin-Token"],
    )


configure_cors(app, guard.SETTINGS.cors_origins)


def _require_admin(request: Request) -> None:
    # 404 rather than 403, so the debug routes do not advertise themselves.
    if not guard.admin_allowed(request.headers.get("x-admin-token")):
        raise HTTPException(404, "Not found")


@app.get("/")
async def index():
    p = BASE_DIR / "index.html"
    if not p.exists():
        return JSONResponse({"service": "Crop Oracle", "status": "running"})
    return FileResponse(p)


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page():
    p = BASE_DIR / "privacy.html"
    if not p.exists():
        raise HTTPException(404, "Not found")
    import html as _html
    if PRIVACY_CONTACT:
        who = f"<p>{_html.escape(PRIVACY_OPERATOR) + '<br>' if PRIVACY_OPERATOR else ''}" \
              f"Contact: {_html.escape(PRIVACY_CONTACT)}</p>"
    else:
        who = ('<div class="warn">The operator of this site has not yet added a contact address. '
               'Until they do, the option to let us keep photos is switched off.</div>')
    kindwise = ("<li>Your photo is also sent to <em>Kindwise (crop.health)</em>, an outside plant-disease "
                "service, for a second opinion. Its hidden camera data (such as GPS) is removed first.</li>"
                if KINDWISE_KEY else "")
    html = (p.read_text(encoding="utf-8")
              .replace("{{RETENTION_DAYS}}", str(FEEDBACK_RETENTION_DAYS))
              .replace("{{CONTACT_BLOCK}}", who)
              .replace("{{KINDWISE_ITEM}}", kindwise))
    return HTMLResponse(html)


def _load_kvk() -> dict:
    try:
        return json.loads(KVK_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print("[kvk] could not load kvk_contacts.json:", repr(e))
        return {"national": {}, "directory_url": "", "districts": {}}


KVK_DATA = _load_kvk()


@app.get("/api/kvk")
async def kvk_contacts():
    """Kisan Call Centre number and the KVKs listed for each Karnataka district."""
    return KVK_DATA


@app.get("/api/health")
async def health(request: Request):
    base = {"ok": True, "version": APP_VERSION, "focus": "Karnataka"}
    if not guard.admin_allowed(request.headers.get("x-admin-token")):
        return base          # the details below stay private
    return {
        **base,
        "supported_crops": list(SUPPORTED_CROPS),
        "diseases_registered": len(DISEASES_BY_ID),
        "remedies_loaded": len([k for k in REMEDIES if not k.startswith("_norm:")]),
        "tflite_classes": len(TFLITE_IDX_TO_CLASS),
        "tflite_mapping": len(TFLITE_CLASS_TO_META),
        "rice_model_loaded": RICE_INTERPRETER is not None,
        "rice_model_classes": RICE_CLASS_NAMES if RICE_INTERPRETER else [],
        "rice_input_scale": RICE_INPUT_SCALE,
        "plantvillage": PLANTVILLAGE.stats() if PLANTVILLAGE.available else {"available": False},
        "plantvillage_stats_loaded": len(PV_STATS),
        "debug_masks_enabled": DEBUG_MASKS,
        "sarvam": SARVAM_ON,
        "kindwise": bool(KINDWISE_KEY),
        "slm": {"enabled": SLM_ENABLED, "model": SLM_MODEL, "provider": "ollama"},
    }


@app.get("/api/remedy/{disease_id}")
async def get_remedy(disease_id: str, crop: str = "rice"):
    d = DISEASES_BY_ID.get(disease_id)
    if not d:
        raise HTTPException(404, f"Unknown disease id '{disease_id}'")
    r = find_remedy(crop, disease_id)
    if not r:
        raise HTTPException(404, f"No remedy registered for '{disease_id}' on crop '{crop}'")
    return r


@app.get("/api/coverage")
async def coverage(request: Request):
    _require_admin(request)
    all_ids = set(DISEASES_BY_ID.keys())
    with_remedy, without_remedy = [], []
    for did, d in DISEASES_BY_ID.items():
        if any(find_remedy(c, did) for c in d["crops"]):
            with_remedy.append(did)
        else:
            without_remedy.append(did)
    tflite_ids = {v.get("id") for v in TFLITE_CLASS_TO_META.values() if v.get("id")}
    tflite_missing = sorted(set(DISEASES_BY_ID.keys()) - tflite_ids)
    pv_counts = {}
    if PLANTVILLAGE.available:
        for cls in TFLITE_CLASS_TO_META.keys():
            pv_counts[cls] = PLANTVILLAGE.count_for(cls)
    return {
        "csv_remedies": {
            "total_diseases": len(all_ids),
            "with_remedy": len(with_remedy),
            "without_remedy": sorted(without_remedy),
            "coverage_pct": round(100 * len(with_remedy) / len(all_ids), 1) if all_ids else 0.0,
        },
        "tflite_mapping": {
            "classes_mapped": len(TFLITE_CLASS_TO_META),
            "diseases_covered": len(tflite_ids),
            "diseases_missing": tflite_missing,
        },
        "rice_model": {
            "loaded": RICE_INTERPRETER is not None,
            "classes": RICE_CLASS_NAMES,
        },
        "plantvillage": {
            "available": PLANTVILLAGE.available,
            "sample_counts": {k: v for k, v in sorted(pv_counts.items()) if v > 0},
            "classes_with_zero_images": sorted(k for k, v in pv_counts.items() if v == 0),
        },
    }


@app.get("/api/dataset/stats")
async def dataset_stats(request: Request):
    _require_admin(request)
    return PLANTVILLAGE.stats()


@app.get("/api/dataset/class/{class_name}")
async def dataset_class(request: Request, class_name: str, n: int = 3):
    _require_admin(request)
    n = max(1, min(n, 20))
    if not PLANTVILLAGE.available:
        raise HTTPException(404, "PlantVillage dataset not installed")
    if class_name not in PLANTVILLAGE.classes:
        raise HTTPException(404, f"Unknown class '{class_name}'")
    return {
        "class": class_name,
        "count": PLANTVILLAGE.count_for(class_name),
        "samples": PLANTVILLAGE.sample_paths(class_name)[:n],
    }


@app.get("/api/dataset/sample-image/{class_name}")
async def dataset_sample_image(request: Request, class_name: str, idx: int = 0):
    _require_admin(request)
    if not PLANTVILLAGE.available:
        raise HTTPException(404, "PlantVillage dataset not installed")
    paths = PLANTVILLAGE.sample_paths(class_name)
    if not paths:
        raise HTTPException(404, f"No samples for '{class_name}'")
    idx = max(0, min(idx, len(paths) - 1))
    return FileResponse(paths[idx], media_type="image/jpeg")


# ============================================================
# PIPELINE STAGE TEXT (localised)
# ============================================================

STAGE_LABELS = {
    "vision": {"en": "Vision agent", "hi": "दृष्टि एजेंट", "kn": "ದೃಷ್ಟಿ ಏಜೆಂಟ್"},
    "weather": {"en": "Weather agent", "hi": "मौसम एजेंट", "kn": "ಹವಾಮಾನ ಏಜೆಂಟ್"},
    "location": {"en": "Location agent", "hi": "स्थान एजेंट", "kn": "ಸ್ಥಳ ಏಜೆಂಟ್"},
    "knowledge": {"en": "Knowledge agent", "hi": "ज्ञान एजेंट", "kn": "ಜ್ಞಾನ ಏಜೆಂಟ್"},
    "regulatory": {"en": "Rules agent", "hi": "नियम एजेंट", "kn": "ನಿಯಮ ಏಜೆಂಟ್"},
    "slm": {"en": "Decision agent", "hi": "निर्णय एजेंट", "kn": "ನಿರ್ಧಾರ ಏಜೆಂಟ್"},
    "farmer": {"en": "Farmer advisor", "hi": "किसान सलाहकार", "kn": "ರೈತ ಸಲಹೆಗಾರ"},
}

STAGE_TEXT = {
    "vision": {
        "en": "{n} spot clusters · brown area {brown}% · yellowing {chl}%",
        "hi": "{n} धब्बों के समूह · भूरा क्षेत्र {brown}% · पीलापन {chl}%",
        "kn": "{n} ಚುಕ್ಕೆ ಗುಂಪುಗಳು · ಕಂದು ಭಾಗ {brown}% · ಹಳದಿ ಭಾಗ {chl}%",
    },
    "vision_model": {
        "en": " · image model: {cls} ({conf}%)",
        "hi": " · छवि मॉडल: {cls} ({conf}%)",
        "kn": " · ಚಿತ್ರ ಮಾದರಿ: {cls} ({conf}%)",
    },
    "vision_local": {
        "en": " · local analysis only", "hi": " · केवल स्थानीय विश्लेषण", "kn": " · ಸ್ಥಳೀಯ ವಿಶ್ಲೇಷಣೆ ಮಾತ್ರ",
    },
    "weather": {
        "en": "{t} °C · {rh}% humidity · {r} mm rain in 48 h · {src}",
        "hi": "{t} °C · {rh}% आर्द्रता · 48 घंटे में {r} मिमी बारिश · {src}",
        "kn": "{t} °C · {rh}% ಆರ್ದ್ರತೆ · 48 ಗಂಟೆಗಳಲ್ಲಿ {r} ಮಿಮೀ ಮಳೆ · {src}",
    },
    "weather_live": {"en": "live data", "hi": "लाइव डेटा", "kn": "ನೇರ ಮಾಹಿತಿ"},
    "weather_est": {"en": "estimate (offline)", "hi": "अनुमान (ऑफ़लाइन)", "kn": "ಅಂದಾಜು (ಆಫ್‌ಲೈನ್)"},
    "outside_ka": {
        "en": " · outside Karnataka", "hi": " · कर्नाटक के बाहर", "kn": " · ಕರ್ನಾಟಕದ ಹೊರಗೆ",
    },
    "knowledge": {
        "en": "{n} evidence sources · {live} live from PubMed",
        "hi": "{n} साक्ष्य स्रोत · {live} PubMed से लाइव",
        "kn": "{n} ಪುರಾವೆ ಮೂಲಗಳು · {live} PubMed ನಿಂದ ನೇರ",
    },
    "knowledge_fallback": {
        "en": " · built-in library only", "hi": " · केवल अंतर्निहित संग्रह", "kn": " · ಅಂತರ್ನಿರ್ಮಿತ ಸಂಗ್ರಹ ಮಾತ್ರ",
    },
    "knowledge_skipped": {
        "en": "skipped (the leaf looks healthy)",
        "hi": "छोड़ा गया (पत्ती स्वस्थ दिख रही है)",
        "kn": "ಬಿಟ್ಟಿದೆ (ಎಲೆ ಆರೋಗ್ಯವಾಗಿ ಕಾಣುತ್ತಿದೆ)",
    },
    "regulatory_sub": {
        "en": "{n} product(s) replaced under {region} rules",
        "hi": "{region} के नियमों के अनुसार {n} उत्पाद बदले गए",
        "kn": "{region} ನಿಯಮಗಳ ಪ್ರಕಾರ {n} ಉತ್ಪನ್ನ ಬದಲಿಸಲಾಗಿದೆ",
    },
    "regulatory_ok": {
        "en": "{region} rules checked", "hi": "{region} के नियम जाँचे गए", "kn": "{region} ನಿಯಮಗಳನ್ನು ಪರಿಶೀಲಿಸಲಾಗಿದೆ",
    },
    "regulatory_none": {
        "en": "no products proposed", "hi": "कोई उत्पाद नहीं सुझाया गया", "kn": "ಯಾವುದೇ ಉತ್ಪನ್ನ ಸೂಚಿಸಿಲ್ಲ",
    },
    "regulatory_withheld": {
        "en": "pesticide advice withheld: the result is not certain",
        "hi": "कीटनाशक सलाह रोकी गई: नतीजा पक्का नहीं है",
        "kn": "ಕೀಟನಾಶಕ ಸಲಹೆ ತಡೆಹಿಡಿಯಲಾಗಿದೆ: ಫಲಿತಾಂಶ ಖಚಿತವಿಲ್ಲ",
    },
    "slm_used": {
        "en": "{model} · evidence-based reasoning",
        "hi": "{model} · साक्ष्य-आधारित तर्क",
        "kn": "{model} · ಪುರಾವೆ-ಆಧಾರಿತ ತರ್ಕ",
    },
    "slm_fallback": {
        "en": "rule-based reasoning (no language model connected)",
        "hi": "नियम-आधारित तर्क (कोई भाषा मॉडल जुड़ा नहीं है)",
        "kn": "ನಿಯಮ-ಆಧಾರಿತ ತರ್ಕ (ಯಾವುದೇ ಭಾಷಾ ಮಾದರಿ ಸಂಪರ್ಕಿತವಿಲ್ಲ)",
    },
    "farmer": {
        "en": "action plan prepared in English",
        "hi": "कार्य योजना हिंदी में तैयार",
        "kn": "ಕ್ರಿಯಾ ಯೋಜನೆ ಕನ್ನಡದಲ್ಲಿ ಸಿದ್ಧ",
    },
}


def _st(key: str, lang: str) -> str:
    entry = STAGE_TEXT[key]
    return entry.get(lang) or entry["en"]


# Without a trained model the answer rests on colour and shape alone, and for most crops several diseases
# look alike in those measures. These cut-offs were set by hand and have not been checked against expert
# labels (see README, "Measuring accuracy"). Do not loosen them to see fewer "low" results: that would
# only make the label say more than the evidence does.
HIGH_MIN_SCORE, HIGH_MIN_GAP = 0.60, 0.15
MODERATE_MIN_SCORE, MODERATE_MIN_GAP = 0.40, 0.08


def decide_confidence(*, is_healthy: bool, model_agrees: bool, model_disagrees: bool,
                      top_score: float, gap: float) -> str:
    """'high', 'moderate' or 'low'. The trained model can raise or flag, never flip, the diagnosis."""
    if is_healthy:
        return "high"
    if model_agrees:
        return "high"
    if model_disagrees:
        return "low"
    if top_score >= HIGH_MIN_SCORE and gap >= HIGH_MIN_GAP:
        return "high"
    if top_score >= MODERATE_MIN_SCORE and gap >= MODERATE_MIN_GAP:
        return "moderate"
    return "low"


def low_confidence_reason(*, inconclusive: bool, severity: str, model_disagrees: bool,
                          top_score: float) -> str:
    """Why a result is 'low', so the page can say the true thing. A good photo must never be blamed
    for a result that is low because two diseases look alike.
      photo  - the photo is too poor to read          weak  - nothing matches this crop's diseases well
      model  - the trained model disagrees            alike - two or more diseases fit about equally well"""
    if inconclusive or severity == "severe":
        return "photo"
    if model_disagrees:
        return "model"
    if top_score < MODERATE_MIN_SCORE:
        return "weak"
    return "alike"


async def _read_upload(upload: UploadFile, limit: int = MAX_IMAGE_BYTES) -> bytes:
    """Read an uploaded file in chunks and stop as soon as it is over the limit."""
    chunks, total = [], 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(413, "Image is too large (max 10 MB).")
        chunks.append(chunk)
    return b"".join(chunks)


async def _acquire_analysis_slot():
    """Wait for one of the CPU slots. Returns the semaphore (or None when limits are off)."""
    sem = ANALYSIS_SEM
    if sem is None:
        return None
    try:
        await asyncio.wait_for(sem.acquire(), timeout=guard.SETTINGS.analysis_queue_seconds)
    except asyncio.TimeoutError:
        raise HTTPException(503, "The server is busy. Please try again in a moment.",
                            headers={"Retry-After": "10"})
    return sem


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    crop: str = Form("rice"),
    region: str = Form("ka"),
    language: str = Form("en"),
    latitude: float = Form(12.9716),
    longitude: float = Form(77.5946),
    symptoms_text: str = Form(""),
    live_pubmed: bool = Form(False),
    tflite_top3: str = Form("[]"),
):
    start = time.perf_counter()
    crop = (crop or "rice").strip().lower()
    region = (region or "ka").strip().lower()
    language = (language or "en").strip().lower()
    if language not in ("en", "hi", "kn"):
        language = "en"
    if region not in REGION_NAMES:
        region = "ka"
    symptoms_text = (symptoms_text or "")[:MAX_SYMPTOMS_CHARS]

    if not (math.isfinite(latitude) and math.isfinite(longitude)
            and -90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        raise HTTPException(400, "Latitude must be between -90 and 90 and longitude between -180 and 180.")
    if crop not in DISEASES_BY_CROP:
        raise HTTPException(400, f"Crop '{crop}' is not supported. Supported crops: {', '.join(SUPPORTED_CROPS)}.")
    inside_ka = in_karnataka(latitude, longitude)

    # Predictions sent by the client are ignored unless ALLOW_CLIENT_ML is on (they can be forged).
    try:
        tflite_preds = json.loads(tflite_top3) if (ALLOW_CLIENT_ML and tflite_top3) else []
        if not isinstance(tflite_preds, list):
            tflite_preds = []
    except Exception:
        tflite_preds = []
    tflite_id, tflite_conf, tflite_evidence = tflite_hint_to_internal(tflite_preds)
    tflite_class_name = tflite_evidence[0]["class"] if tflite_evidence else None

    raw = await _read_upload(image)
    if not raw:
        raise HTTPException(400, "Image is empty.")

    # Decode, measure the leaf and run the rice model while holding one of a few CPU slots,
    # so a burst of uploads cannot use up all the memory.
    slot = await _acquire_analysis_slot()
    try:
        try:
            pil = await asyncio.to_thread(_decode_image, raw)
        except Exception as e:
            print("[image] failed:", repr(e))
            raise HTTPException(400, "Invalid image file.")
        try:
            features = await asyncio.to_thread(analyze_leaf, pil, True)
            if KINDWISE_KEY:
                # What goes to the outside service is re-encoded, so phone GPS/EXIF data is not sent.
                raw = await asyncio.to_thread(fb.encode_photo, pil, 1600)
            if not tflite_preds and crop == "rice":
                auto_preds = await asyncio.to_thread(_run_rice_model, pil)
                if auto_preds:
                    tflite_preds = auto_preds
                    tflite_id, tflite_conf, tflite_evidence = tflite_hint_to_internal(tflite_preds)
                    tflite_class_name = tflite_evidence[0]["class"] if tflite_evidence else None
                    print(f"[tflite] auto-run top: {auto_preds[0]['class']} "
                          f"({auto_preds[0]['confidence']*100:.1f}%)")
        except Exception as e:
            print("[vision] failed:", repr(e))
            raise HTTPException(500, "Image analysis failed.")
        finally:
            pil.close()
    finally:
        if slot is not None:
            slot.release()

    extras = features.pop("_extras")
    photo_quality = extras["quality"]
    overlay = extras["overlay"]
    severity = photo_quality["severity"]

    # A photo with (almost) no leaf in it must never be reported as "healthy".
    if features.get("leaf_fraction", 1.0) < MIN_LEAF_FRACTION:
        # "detail" stays a plain sentence. "code" and "photo_quality" let the page say *why*
        # (too dark, blurry, ...) in the farmer's language.
        return JSONResponse({
            "detail": ("No leaf could be found in this photo. Retake it in daylight with a single "
                       "leaf filling most of the frame, against a plain background."),
            "code": "no_leaf",
            "photo_quality": photo_quality,
        }, status_code=422)

    # Parallel external calls
    kindwise, weather, place = await asyncio.gather(
        kindwise_analyze(raw, crop, language),
        fetch_weather_cached(latitude, longitude),
        reverse_geocode_cached(latitude, longitude),
    )

    kindwise_id = None
    kindwise_conf = 0.0
    if kindwise:
        kid = kindwise.get("disease_id")
        kconf = float(kindwise.get("confidence", 0.0))
        if kid and kid in VALID_DISEASE_IDS and kconf >= 0.40:
            kindwise_id, kindwise_conf = kid, kconf

    # Let the model contribute to ranking if it's reasonably confident
    rank_ml_id = None
    rank_ml_conf = 0.0
    if tflite_id and tflite_conf >= 0.55:
        rank_ml_id = tflite_id
        rank_ml_conf = tflite_conf

    scored = rank_candidates(
        features, crop, weather, symptoms_text,
        rank_ml_id, rank_ml_conf, tflite_class_name,
        kindwise_id, kindwise_conf,
    )
    if not scored:
        raise HTTPException(400, f"No diseases registered for crop '{crop}'.")

    top = scored[0]
    second = scored[1] if len(scored) > 1 else None

    looks_healthy = (
        top["vision_score"] < 0.35
        and top["score"] < 0.38
        and not (kindwise_id and kindwise_conf >= 0.70)
    )
    # "No disease found" in a photo that is too poor to see disease in is not a result.
    # Say so, instead of telling the farmer the crop is healthy.
    inconclusive = looks_healthy and severity == "severe"
    is_healthy = looks_healthy and not inconclusive

    # Confidence: model can boost or flag, but never flip the diagnosis
    model_agrees = bool(tflite_id and tflite_id == top["id"] and tflite_conf >= 0.55)
    model_disagrees = bool(tflite_id and tflite_id != top["id"] and tflite_conf >= 0.85)

    gap = top["score"] - (second["score"] if second else 0.0)
    confidence = decide_confidence(
        is_healthy=looks_healthy, model_agrees=model_agrees, model_disagrees=model_disagrees,
        top_score=top["score"], gap=gap)
    # A poor photo can only lower the confidence.
    confidence = quality.cap_confidence(confidence, severity, looks_healthy)
    if inconclusive:
        confidence = "low"

    # Uncertain diagnosis: show no spray or fertiliser steps (see apply_input_gate).
    spray_withheld = (not looks_healthy) and confidence == "low"
    confidence_reason = (low_confidence_reason(inconclusive=inconclusive, severity=severity,
                                               model_disagrees=model_disagrees, top_score=top["score"])
                         if (inconclusive or spray_withheld) else None)

    actual_region = region  # the rules the farmer selected always apply

    if looks_healthy:
        citations, live_count, live_failed = [], 0, False
    else:
        citations, live_count, live_failed = await retrieve(top, live_pubmed)

    top_pv_class = next(
        (e.get("class") for e in tflite_evidence if e.get("internal_id") == top["id"]),
        None,
    )
    remedy = find_remedy(crop, top["id"], top_pv_class)

    top_n = scored[:TOP_N_CANDIDATES] if not looks_healthy else []
    candidate_cards = []
    for idx, cand in enumerate(top_n):
        cand_reg = apply_input_gate(
            check_regulation(cand, actual_region, lang=language), spray_withheld)
        cand_plan = build_plan(cand, cand_reg, cand["refs"], CORPUS, lang=language)
        cand_labels = {
            b: BUCKET_LABELS[b].get(language) or BUCKET_LABELS[b]["en"]
            for b in cand_plan
        }
        name_local = cand["name"].get(language) or cand["name"]["en"]
        display = dict(cand)
        names = dict(cand["name"])
        names[language] = name_local
        display["name"] = names

        cits = citations
        conflicts = render_conflicts(
            find_conflicts(cand, second, weather, symptoms_text,
                           cand_reg, features, actual_region, lang=language),
            language,
        )

        candidate_cards.append({
            "rank": idx,
            "diagnosis": _strip_disease(display),
            "is_top": idx == 0,
            "regulatory": cand_reg,
            "plan": {"buckets": cand_plan, "labels": cand_labels},
            "citations": cits,
            "conflicts": conflicts,
            "spoken": build_spoken(cand, is_healthy, cand_plan, language, name_local,
                                   uncertain=spray_withheld),
            # The treatment notes in the CSV name products, so they are hidden too.
            "csv_remedy": remedy if (idx == 0 and not spray_withheld) else None,
        })

    if inconclusive:
        slm = {"enabled": SLM_ENABLED, "used": False, "model": SLM_MODEL, "mode": "skipped",
               "reasoning": _fallback_reasoning(top, second, features, weather, symptoms_text,
                                                True, language)}
        health_score = None
    else:
        slm = await run_slm_agent(top, second, features, weather, symptoms_text,
                                  is_healthy, language)
        health_score = build_health_score(top, weather, features, is_healthy, language)

    # Build the stage summary shown in the pipeline UI, in the farmer's language
    stage_detail_vision = _st("vision", language).format(
        n=features["spot_count"],
        brown=f"{features['lesion_ratio'] * 100:.1f}",
        chl=f"{features['chlorosis_ratio'] * 100:.1f}",
    )
    if tflite_evidence:
        stage_detail_vision += _st("vision_model", language).format(
            cls=tflite_evidence[0]["class"], conf=f"{tflite_evidence[0]['confidence'] * 100:.0f}")
    else:
        stage_detail_vision += _st("vision_local", language)

    weather_detail = _st("weather", language).format(
        t=f"{weather['temp']:.1f}", rh=round(weather["rh"]), r=f"{weather['rain_48h']:.1f}",
        src=_st("weather_live" if weather["live"] else "weather_est", language))

    location_detail = (place.get("label") if place and place.get("label")
                       else "{:.4f}, {:.4f}".format(latitude, longitude))
    if not inside_ka:
        location_detail += _st("outside_ka", language)

    if looks_healthy:
        knowledge_detail = _st("knowledge_skipped", language)
    else:
        knowledge_detail = _st("knowledge", language).format(n=len(citations), live=live_count)
        if live_failed:
            knowledge_detail += _st("knowledge_fallback", language)

    region_name = _region_label(actual_region, language)
    top_reg = candidate_cards[0]["regulatory"] if candidate_cards else None
    if looks_healthy:
        regulatory_detail = _st("regulatory_none", language)
    elif spray_withheld:
        regulatory_detail = _st("regulatory_withheld", language)
    elif top_reg and top_reg["substitutions"]:
        regulatory_detail = _st("regulatory_sub", language).format(
            n=len(top_reg["substitutions"]), region=region_name)
    else:
        regulatory_detail = _st("regulatory_ok", language).format(region=region_name)

    slm_detail = (_st("slm_used", language).format(model=slm["model"]) if slm["used"]
                  else _st("slm_fallback", language))

    stages = [
        {"id": sid, "label": STAGE_LABELS[sid].get(language) or STAGE_LABELS[sid]["en"], "detail": detail}
        for sid, detail in (
            ("vision", stage_detail_vision),
            ("weather", weather_detail),
            ("location", location_detail),
            ("knowledge", knowledge_detail),
            ("regulatory", regulatory_detail),
            ("slm", slm_detail),
            ("farmer", _st("farmer", language)),
        )
    ]

    spoken = (candidate_cards[0]["spoken"] if candidate_cards
              else build_spoken(top, True, {}, language, inconclusive=inconclusive))

    # The page sends this token back with the farmer's "was this right?" answer. Signing it
    # means the answer is stored against what the app really said.
    analysis_id = uuid.uuid4().hex
    kind = "inconclusive" if inconclusive else "healthy" if is_healthy else "diagnosis"
    analysis_token = STORE.sign_analysis({
        "id": analysis_id, "crop": crop, "kind": kind,
        "pred": None if inconclusive else ("healthy" if is_healthy else top["id"]),
        "conf": confidence, "score": round(top["score"], 3),
        "q": [i["code"] for i in photo_quality["issues"]],
    })

    return {
        "elapsed_ms": round((time.perf_counter() - start) * 1000),
        "version": APP_VERSION,
        "language": language,
        "crop": crop,
        "region": actual_region,
        "in_karnataka": inside_ka,
        "is_healthy": is_healthy,
        "inconclusive": inconclusive,
        "spray_withheld": spray_withheld,
        "confidence": confidence,
        "confidence_reason": confidence_reason,
        # The disease that scored nearly the same as the answer, so the page can name both.
        "runner_up": ({"id": second["id"], "name": dict(second["name"])}
                      if (confidence_reason == "alike" and second) else None),
        "spoken": spoken,
        "candidates": candidate_cards,
        "health_score": health_score,
        "slm": slm,
        "stages": stages,
        "vision": features,
        "vision_overlay": overlay,
        "photo_quality": photo_quality,
        "analysis_id": analysis_id,
        "analysis_token": analysis_token,
        "photo_feedback_enabled": bool(PRIVACY_CONTACT),
        "diagnosis_options": [{"id": d["id"], "name": d["name"]} for d in DISEASES_BY_CROP[crop]],
        "weather": weather,
        "place": place,
        "evidence_sources": {
            "tflite": tflite_evidence,
            "rice_model_used": bool(tflite_preds),
            "kindwise": ({"id": kindwise_id, "name": kindwise.get("disease_name"),
                          "confidence": kindwise_conf} if kindwise_id else None),
            "csv_remedy_found": remedy is not None,
        },
        "ui_strings": T.get(language, T["en"]),
    }


def _strip_disease(d: dict) -> Optional[dict]:
    if not d:
        return None
    return {
        "id": d["id"],
        "name": d["name"],
        "pathogen": d["pathogen"],
        "score": round(d["score"], 4),
        "vision_score": round(d["vision_score"], 4),
        "weather_suit": round(d["weather_suit"], 4),
        "symptom_score": round(d["symptom_score"], 4),
        "ml_boost": round(d.get("ml_boost", 0.0), 4),
        "envelope": d["envelope"],
    }


# ============================================================
# FEEDBACK  ("was this result right?")
# ============================================================

FEEDBACK_OTHER_IDS = ("healthy", "other")


@app.post("/api/feedback")
async def submit_feedback(
    analysis_token: str = Form(...),
    verdict: str = Form(...),
    reported_id: str = Form(""),
    consent_photo: bool = Form(False),
    language: str = Form("en"),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    image: Optional[UploadFile] = File(None),
):
    """Store the farmer's answer. The photo and rounded location are kept only when the
    farmer ticked the consent box. Returns a code the farmer can use to delete the record."""
    analysis = STORE.verify_analysis(analysis_token)
    if not analysis:
        raise HTTPException(400, "This result is too old or not valid. Please analyse the photo again.")
    verdict = (verdict or "").strip().lower()
    if verdict not in fb.VERDICTS:
        raise HTTPException(400, "verdict must be one of: " + ", ".join(fb.VERDICTS))

    reported = None
    if verdict == "different":
        reported = (reported_id or "").strip()
        allowed = {d["id"] for d in DISEASES_BY_CROP.get(analysis["crop"], [])} | set(FEEDBACK_OTHER_IDS)
        if reported not in allowed:
            raise HTTPException(400, "reported_id is not a known disease for this crop.")

    language = (language or "en").strip().lower()
    if language not in ("en", "hi", "kn"):
        language = "en"

    # Photos are kept only when the operator has published a privacy contact.
    consent_photo = bool(consent_photo) and bool(PRIVACY_CONTACT)

    photo_jpeg = None
    if consent_photo:
        if image is None:
            raise HTTPException(400, "consent_photo is set but no photo was sent.")
        raw = await _read_upload(image)
        if not raw:
            raise HTTPException(400, "Image is empty.")
        try:
            pil = await asyncio.to_thread(_decode_image, raw)
            photo_jpeg = await asyncio.to_thread(fb.encode_photo, pil)
            pil.close()
        except Exception as e:
            print("[feedback] image failed:", repr(e))
            raise HTTPException(400, "Invalid image file.")

    lat = lon = None
    if (consent_photo and latitude is not None and longitude is not None
            and math.isfinite(latitude) and math.isfinite(longitude)
            and -90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        lat, lon = latitude, longitude

    try:
        fid, delete_token = await asyncio.to_thread(
            lambda: STORE.add(analysis=analysis, verdict=verdict, reported_id=reported,
                              consent_photo=bool(consent_photo), photo_jpeg=photo_jpeg,
                              lat=lat, lon=lon, language=language, app_version=APP_VERSION))
    except FileExistsError:
        raise HTTPException(409, "You have already answered for this result.")
    return {"ok": True, "id": fid, "delete_token": delete_token, "photo_saved": bool(photo_jpeg)}


@app.delete("/api/feedback/{feedback_id}")
async def delete_feedback(request: Request, feedback_id: str):
    token = request.headers.get("x-delete-token", "")
    outcome = await asyncio.to_thread(STORE.delete, feedback_id, token)
    if outcome == "deleted":
        return {"ok": True}
    # The same answer for "no such record" and "wrong code" so ids cannot be probed.
    raise HTTPException(404, "No record with that id and code.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)