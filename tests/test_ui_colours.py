"""Readable colours in both themes, checked straight from the page's colour tokens.

The page keeps every colour in `:root{...}` (light) and `:root[data-theme="dark"]{...}` (dark).
Each pair below is text-on-background that the page really uses; the numbers are WCAG contrast ratios
(4.5 is the level for normal text, 3 for large text and outlines)."""
import re

import main

PAGE = (main.BASE_DIR / "index.html").read_text(encoding="utf-8")


def block(start_marker):
    a = PAGE.index(start_marker)
    return PAGE[a:PAGE.index("\n}", a)]


def tokens(css):
    return {k: v.strip() for k, v in re.findall(r"(--[a-z0-9-]+):\s*([^;]+);", css)}


LIGHT = tokens(block(":root{\n  color-scheme:light;"))
DARK = {**LIGHT, **tokens(block(':root[data-theme="dark"]{'))}


def rgb(value):
    value = value.strip()
    if value in ("#fff", "white"):
        value = "#ffffff"
    m = re.fullmatch(r"#([0-9a-fA-F]{6})", value)
    assert m, f"not a plain hex colour: {value!r}"
    h = m.group(1)
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def luminance(c):
    def lin(x):
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4
    r, g, b = map(lin, c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(fg, bg):
    a, b = luminance(rgb(fg)), luminance(rgb(bg))
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def resolve(theme, name):
    return name if name.startswith("#") else theme[f"--{name}"]


# (text token or hex, background tokens, minimum ratio)
TEXT_ON = [
    ("txt", ["bg", "card", "bg-2", "field", "raise", "accent-soft", "ok-a", "ok-b", "info-a", "info-b", "warn-a", "warn-b", "ai-a", "ai-b", "run", "hero", "drop-a", "drop-b", "guide-bg"], 7),
    ("txt-2", ["bg", "card", "bg-2", "accent-soft", "ai-a", "ai-b"], 4.5),
    ("mut", ["bg", "card", "bg-2", "field", "ok-a", "ok-b", "info-a", "info-b", "warn-a", "warn-b", "ai-a", "ai-b", "drop-a", "drop-b", "guide-bg"], 4.5),
    ("accent-d", ["bg", "card", "bg-2", "accent-soft", "ok-a", "ok-b", "run"], 4.5),
    ("gold-ink", ["gold-soft", "warn-a", "warn-b", "bg", "card"], 4.5),
    ("red-ink", ["red-soft", "card"], 4.5),
    ("blue-ink", ["blue-soft", "info-a", "info-b"], 4.5),
    ("blue", ["blue-soft"], 4.5),
    ("purple", ["purple-soft"], 4.5),
    ("bd-high-t", ["bd-high"], 4.5),
    ("bd-mod-t", ["bd-mod"], 4.5),
    ("bd-low-t", ["bd-low"], 4.5),
    ("#ffffff", ["fill", "fill-d", "toast-bg"], 4.5),
]
OUTLINES = [("accent", ["card", "bg"], 3), ("focus", ["card", "bg", "field"], 3), ("field-line", ["field"], 3)]


def check(theme, name):
    problems = []
    for fg, backgrounds, need in TEXT_ON + OUTLINES:
        for bg in backgrounds:
            got = contrast(resolve(theme, fg), resolve(theme, bg))
            if got < need:
                problems.append(f"{name}: {fg} on {bg} is {got:.2f}, needs {need}")
    return problems


def test_light_theme_is_readable():
    assert not check(LIGHT, "light")


def test_dark_theme_is_readable():
    assert not check(DARK, "dark")


def test_dark_theme_is_really_dark():
    for name in ("bg", "card", "bg-2", "field", "ok-a", "warn-a", "info-a", "ai-a", "drop-a", "hero"):
        assert luminance(rgb(DARK[f"--{name}"])) < 0.08, name
    assert luminance(rgb(DARK["--txt"])) > 0.7


def test_dark_theme_redefines_every_colour_that_would_stay_bright():
    """A colour token left at its light value would shine out of a dark page."""
    bright_in_light = [k for k, v in LIGHT.items()
                       if re.fullmatch(r"#[0-9a-fA-F]{6}", v) and luminance(rgb(v)) > 0.55]
    missing = [k for k in bright_in_light if k not in tokens(block(':root[data-theme="dark"]{'))]
    # gold and the camera/fill colours are meant to stay the same in both themes
    assert set(missing) <= {"--gold", "--purple-soft"} or not missing, missing


def test_the_page_has_no_stray_light_colours_outside_the_tokens():
    """Colours written straight into a rule can not follow the theme. Only a short list of deliberate ones is allowed."""
    css = PAGE[PAGE.index("<style>"):PAGE.index("</style>")]
    css = css.replace(block(":root{\n  color-scheme:light;"), "").replace(block(':root[data-theme="dark"]{'), "")
    stray = []
    for line in css.split("\n"):
        selector = line.split("{")[0].strip()
        for colour in re.findall(r"#[0-9a-fA-F]{3,8}\b", line):
            if colour.lower() in ("#fff", "#ffffff", "#000"):
                continue                                   # white text on solid fills; the camera screen
            stray.append((selector, colour))
    deliberate = ("logo", "stripe", "health-card", "summary-hero", "toast.error", "sw", "cam", "health-meter")
    bad = [(s, c) for s, c in stray if not any(d in s for d in deliberate)]
    assert not bad, bad
