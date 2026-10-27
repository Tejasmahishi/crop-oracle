# Crop Oracle Karnataka. One worker, ordinary user, data in /data.
#   docker build -t crop-oracle .
#   docker run -d --name crop-oracle -p 8000:8000 -v crop-oracle-data:/data --env-file .env crop-oracle
# Put HTTPS in front of it (nginx, Caddy or your platform's router); see README.md.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data

WORKDIR /app

COPY requirements.txt .
# ai-edge-litert runs the rice model on Python 3.13. Remove it if you do not use the model.
RUN pip install -r requirements.txt ai-edge-litert

COPY main.py quality.py feedback.py guard.py index.html privacy.html kvk_contacts.json ./
# The rice model and the files that say what its answers mean. Delete this line if you do not have them
# (the app then runs without the model), or mount them read-only at /app/<name> instead.
COPY rice_model.tflite rice_class_indices.json class_indices.json class_indices_mapping.json ./
# Other optional files the app uses if they are next to main.py: agronomy_remedies.csv, plantvillage_stats.json.
# Add a COPY line for each one you have.

RUN useradd --system --uid 10001 --home-dir /app appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /data /app
USER appuser

VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

# One worker: rate limits and the analysis queue are counted per process (see README).
# --no-proxy-headers: the app reads X-Forwarded-For itself, and only as many proxies as TRUSTED_PROXY_COUNT.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-proxy-headers"]
