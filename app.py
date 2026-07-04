"""
Goatface Sprite Console — backend
=================================
One file. It does four jobs:
  1. Detects the GPU backend (ROCm / CUDA / MPS / CPU) at startup.
  2. Loads a selectable pixel-art engine (SDXL+LCM, or FLUX schnell) in the background.
  3. Exposes POST /generate — takes a prompt, returns finished sprite PNGs.
  4. Serves index.html (the GUI) at /.

Post-processing baked in: every image is downscaled to a true pixel grid
(NEAREST, so no blur) and quantized to a limited palette — the two steps
that make diffusion output look like real pixel art instead of a photo
of pixel art.

Run it:   python app.py
Open:     http://localhost:8000   (or the pod's proxy URL on RunPod)
"""

import gc
import io
import os
import time
import threading
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# ---------------------------------------------------------------- settings

TRIGGER = "pixel art"                # prepended to every prompt
NEGATIVE = "blurry, photo, realistic, 3d render, jpeg artifacts, watermark, text"

# --- engine A: Pixel XL — SDXL + LCM LoRA + pixel-art LoRA (fast, few steps) ---
SDXL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdxl"   # few-step sampling LoRA
PIXEL_LORA_ID = "nerijs/pixel-art-xl"              # pixel-art fine-tune

# --- engine B: Flux Pixel — FLUX.1-schnell + a modern pixel-art LoRA (quality) ---
FLUX_ID = "black-forest-labs/FLUX.1-schnell"
FLUX_LORA_ID = "UmeAiRT/FLUX.1-dev-LoRA-Modern_Pixel_art"

# Selectable engines. Each carries the few-step sampler settings from its model
# card. "flux" is guidance-distilled, so it skips the negative prompt and runs
# at guidance 0. The generate handler reads steps/guidance/flux from here.
ENGINES = {
    "pixel_xl": {"label": "Pixel XL (fast)",    "steps": 8, "guidance": 1.5, "flux": False},
    "flux":     {"label": "Flux Pixel (quality)", "steps": 4, "guidance": 0.0, "flux": True},
}
DEFAULT_ENGINE = "pixel_xl"

# Valid downscale targets. "native" (handled separately) skips the downscale
# entirely and quantizes the full 1024px render — for hero assets and mockups.
GRID_SIZES = (32, 64, 96, 128, 192, 256)

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Goatface Sprite Console")

# ---------------------------------------------------------------- shared state
# Everything the request handlers and the background loader thread need to
# agree on lives in this one dict. status is: "loading" | "ready" | "error".

state = {
    "status": "loading",
    "error": None,
    "pipe": None,
    # Which engine is active, and what device preference to (re)load it on.
    "engine": DEFAULT_ENGINE,
    "engine_label": ENGINES[DEFAULT_ENGINE]["label"],
    "device_pref": "auto",    # "auto" | "cuda" | "cpu" — remembered across reloads
    # Hardware facts, filled in by detect_backend() and shown in the header.
    "backend": "CPU",         # "ROCm" | "CUDA" | "MPS" | "CPU"
    "device": "cpu",          # torch device string
    "device_name": "CPU",     # human-readable card name
    "vram_gb": None,          # total VRAM in GB, when we can measure it
}

gen_lock = threading.Lock()   # one generation at a time — keeps VRAM sane


# ---------------------------------------------------------------- backend detect

def detect_backend(prefer="auto"):
    """Work out which compute backend to use.

    prefer is "auto" | "cuda" | "cpu". "auto" picks the best thing available;
    "cpu" forces the slow-but-always-works path. Returns nothing — it writes
    the results straight into state so /health can report them.

    Both ROCm (AMD) and CUDA (NVIDIA) report through torch's "cuda" device API.
    We tell them apart by which build flag torch was compiled with:
      * torch.version.hip  is set on ROCm builds
      * torch.version.cuda is set on CUDA builds
    """
    want_cpu = (prefer == "cpu")

    # Real GPU path (unless CPU was explicitly requested).
    if not want_cpu and torch.cuda.is_available():
        if getattr(torch.version, "hip", None):
            backend = "ROCm"
        else:
            backend = "CUDA"

        name = torch.cuda.get_device_name(0)

        # Total VRAM, bytes -> GB. Best-effort; some drivers won't answer.
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            vram = round(total / (1024 ** 3), 1)
        except Exception:
            vram = None

        state.update(backend=backend, device="cuda", device_name=name, vram_gb=vram)
        return

    # Apple Silicon path. MPS has no clean VRAM query, so leave it None.
    if not want_cpu and torch.backends.mps.is_available():
        state.update(backend="MPS", device="mps", device_name="Apple Silicon (MPS)",
                     vram_gb=None)
        return

    # Nothing accelerated (or CPU was requested). Works, but expect *minutes*
    # per image rather than seconds.
    state.update(backend="CPU", device="cpu", device_name="CPU", vram_gb=None)


# ---------------------------------------------------------------- model load
# The model loads in a background thread so the GUI comes up instantly and can
# show "warming up" instead of hanging. Only one engine is held in memory at a
# time; switching engines drops the old pipeline and frees its VRAM first.

def _unload_pipe():
    """Drop the current pipeline and hand its VRAM back to the driver so a
    different engine has room to load: clear the reference, `del` the object,
    force a GC pass, then empty the CUDA cache."""
    pipe = state["pipe"]
    state["pipe"] = None
    del pipe
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def build_pixel_xl(device, dtype, on_gpu):
    """Pixel XL: SDXL with the LCM LoRA (for few-step sampling) stacked on the
    pixel-art LoRA, weighted and run through the LCM scheduler — straight off
    the nerijs/pixel-art-xl model card."""
    from diffusers import StableDiffusionXLPipeline, LCMScheduler

    token = os.environ.get("HF_TOKEN")   # keep gated downloads working
    if on_gpu:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_ID, torch_dtype=dtype, variant="fp16", use_safetensors=True, token=token,
        )
    else:
        # No fp16 variant on CPU — load the full-precision weights.
        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_ID, torch_dtype=dtype, use_safetensors=True, token=token,
        )

    pipe.load_lora_weights(LCM_LORA_ID, adapter_name="lcm")
    pipe.load_lora_weights(PIXEL_LORA_ID, adapter_name="pixel")
    pipe.set_adapters(["lcm", "pixel"], adapter_weights=[1.0, 1.2])
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    pipe.to(device)
    if on_gpu:
        pipe.enable_vae_tiling()   # cheap VRAM saver, GPU-only
    return pipe


def build_flux(device, dtype, on_gpu):
    """Flux Pixel: FLUX.1-schnell (a distilled, guidance-free model) with a
    modern pixel-art LoRA. That LoRA targets FLUX.1-dev, so it may not load onto
    schnell — if it doesn't, we raise a clear error that /health can show rather
    than letting the loader thread die silently."""
    from diffusers import FluxPipeline

    token = os.environ.get("HF_TOKEN")   # keep gated downloads working
    pipe = FluxPipeline.from_pretrained(FLUX_ID, torch_dtype=dtype, token=token)

    try:
        pipe.load_lora_weights(FLUX_LORA_ID)
    except Exception as e:
        raise RuntimeError(
            f"Flux pixel-art LoRA failed to load on FLUX.1-schnell: {e}. "
            "This LoRA is trained for FLUX.1-dev and may be incompatible with "
            "the distilled schnell checkpoint — try Pixel XL instead."
        )

    if on_gpu:
        # Flux is too big to sit resident in 24GB. Stream components to the GPU
        # on demand instead of a static .to("cuda") — avoids the CUDA OOM.
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
    else:
        pipe.to(device)
    return pipe


def reload_pipeline():
    """(Re)build the active engine on the preferred device, in place. Reads the
    engine + device preference from state, so /engine and /backend just update
    those and call this on a background thread."""
    state["status"] = "loading"
    state["error"] = None
    _unload_pipe()   # free the old engine's VRAM before loading the new one

    try:
        detect_backend(state["device_pref"])
        device = state["device"]
        on_gpu = device in ("cuda", "mps")

        engine = state["engine"]
        state["engine_label"] = ENGINES[engine]["label"]

        if ENGINES[engine]["flux"]:
            # Flux prefers bfloat16 on GPU; fp16 misbehaves on it.
            dtype = torch.bfloat16 if on_gpu else torch.float32
            pipe = build_flux(device, dtype, on_gpu)
        else:
            # fp16 weights on GPU keep VRAM down; CPU needs fp32 for correctness.
            dtype = torch.float16 if on_gpu else torch.float32
            pipe = build_pixel_xl(device, dtype, on_gpu)

        state["pipe"] = pipe
        state["status"] = "ready"
    except Exception as e:  # surface load failures to the GUI via /health
        _unload_pipe()
        state["status"] = "error"
        state["error"] = str(e)


@app.on_event("startup")
def startup():
    """Kick off the first load in the background so the GUI is instant."""
    threading.Thread(target=reload_pipeline, daemon=True).start()


# ---------------------------------------------------------------- pixel post

def pixelate(img: Image.Image, grid, colors: int) -> Image.Image:
    """Turn raw 1024x1024 diffusion output into honest pixel art:
    shrink to the target grid (this *is* the pixelation), then
    quantize to a limited palette like a real sprite sheet.

    grid is either an int size or the string "native". "native" keeps the full
    1024px render and only quantizes the palette — no downscale."""
    if grid != "native":
        img = img.resize((grid, grid), Image.NEAREST)
    return img.quantize(colors=colors, method=Image.MEDIANCUT).convert("RGB")


def resolve_grid(value):
    """Map a requested grid to a valid one: "native" passes through, a known
    size passes through, anything else falls back to the 96 default."""
    if value == "native":
        return "native"
    try:
        g = int(value)
    except (TypeError, ValueError):
        return 96
    return g if g in GRID_SIZES else 96


# ---------------------------------------------------------------- api

class GenRequest(BaseModel):
    prompt: str
    grid: int | str = 96    # 32/64/96/128/192/256, or "native" (no downscale)
    colors: int = 24        # palette size after quantization
    count: int = 4          # how many variations
    seed: int | None = None # set for reproducible results
    vibrance: bool = False  # richer, more saturated palette when True


class BackendRequest(BaseModel):
    device: str = "auto"    # "auto" | "cuda" | "cpu"


class EngineRequest(BaseModel):
    engine: str = DEFAULT_ENGINE   # "pixel_xl" | "flux"


@app.get("/health")
def health():
    return {
        "status": state["status"],
        "error": state["error"],
        "backend": state["backend"],
        "device_name": state["device_name"],
        "vram_gb": state["vram_gb"],
        "engine": state["engine"],
        "engine_label": state["engine_label"],
    }


@app.post("/backend")
def set_backend(req: BackendRequest):
    """Reload the active engine on a different device.

    Returns immediately; the swap happens on a background thread and the status
    goes back to "loading" while it warms up.
    """
    device = req.device if req.device in ("auto", "cuda", "cpu") else "auto"
    state["device_pref"] = device
    threading.Thread(target=reload_pipeline, daemon=True).start()
    return {"status": "loading", "device": device}


@app.post("/engine")
def set_engine(req: EngineRequest):
    """Switch engines. Unloads the current pipeline (freeing VRAM) and loads the
    requested one in the background; status goes back to "loading" during the
    swap.
    """
    engine = req.engine if req.engine in ENGINES else DEFAULT_ENGINE
    state["engine"] = engine
    state["engine_label"] = ENGINES[engine]["label"]
    threading.Thread(target=reload_pipeline, daemon=True).start()
    return {"status": "loading", "engine": engine, "engine_label": ENGINES[engine]["label"]}


@app.post("/generate")
def generate(req: GenRequest):
    if state["status"] != "ready" or state["pipe"] is None:
        return JSONResponse({"error": f"model not ready ({state['status']})"}, 503)

    # Clamp everything so a bad request can't ask for a 900-color, 50-image job.
    req.count = max(1, min(req.count, 8))
    req.colors = max(4, min(req.colors, 64))

    # Resolve the grid: "native" skips the downscale, otherwise it must be one
    # of the known sizes (anything else falls back to the 96 default).
    grid = resolve_grid(req.grid)

    # Vibrance nudges the model toward a richer palette: extra keywords on the
    # positive side, and "muted, desaturated" pushed onto the negative side.
    prompt = f"{TRIGGER}, {req.prompt}"
    negative = NEGATIVE
    if req.vibrance:
        prompt += ", vibrant colors, rich saturated palette, detailed shading"
        negative += ", muted, desaturated"

    # Sampler settings come from the active engine. Flux is guidance-distilled
    # and takes no negative prompt, so we branch on it below.
    cfg = ENGINES[state["engine"]]
    steps, guidance, is_flux = cfg["steps"], cfg["guidance"], cfg["flux"]

    device = state["device"]
    results = []
    with gen_lock:
        pipe = state["pipe"]
        for i in range(req.count):
            # Each variation gets its own seed. A pinned seed steps per image so
            # the batch is reproducible but not four identical sprites.
            seed = (req.seed + i) if req.seed is not None else torch.seed() % (2**31)
            g = torch.Generator(device=device).manual_seed(seed)

            kwargs = dict(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=1024, height=1024,
                generator=g,
            )
            if not is_flux:
                kwargs["negative_prompt"] = negative  # Flux has no negative prompt

            image = pipe(**kwargs).images[0]

            sprite = pixelate(image, grid, req.colors)

            name = f"{int(time.time())}_{seed}.png"
            path = OUT_DIR / name
            # Stash the prompt + seed inside the PNG so files stay self-describing.
            meta = PngInfo()
            meta.add_text("prompt", prompt)
            meta.add_text("seed", str(seed))
            sprite.save(path, pnginfo=meta)

            results.append({"file": f"/outputs/{name}", "seed": seed})

    return results


@app.post("/pixelate")
async def pixelate_upload(
    file: UploadFile = File(...),
    grid: str = Form("96"),
    colors: int = Form(24),
):
    """Run an already-made image (e.g. a Google Flow render) through the same
    post-processing as /generate — resize to the grid with NEAREST, quantize to
    the palette — with no model involved. Works on any machine, even with no
    engine loaded, so it never touches the pipeline or the generation lock."""
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        return JSONResponse({"error": f"could not read image: {e}"}, 400)

    grid = resolve_grid(grid)
    colors = max(4, min(int(colors), 64))
    sprite = pixelate(img, grid, colors)

    # Millisecond stamp so rapid uploads don't collide; "upload" where a seed
    # would normally go, so these read as non-generated in the gallery.
    name = f"{int(time.time() * 1000)}_upload.png"
    path = OUT_DIR / name
    meta = PngInfo()
    meta.add_text("prompt", "uploaded image")
    meta.add_text("source", "upload")
    sprite.save(path, pnginfo=meta)

    return {"file": f"/outputs/{name}", "seed": "upload"}


@app.get("/prompts")
def prompts():
    """Return the prompt library from an optional prompts.txt in the project
    root. One prompt per line; blank lines and #-comments are skipped. Missing
    file just means an empty library — not an error."""
    path = Path("prompts.txt")
    lines = []
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return {"prompts": lines}


@app.get("/gallery")
def gallery():
    """List sprites already sitting in outputs/, newest first (max 60), so a
    page refresh doesn't lose the session's work. Seed is parsed back out of
    the timestamp_seed.png filename."""
    files = sorted(OUT_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    results = []
    for p in files[:60]:
        # Filenames look like "1720000000_12345.png" — the seed is the last
        # underscore-separated chunk of the stem.
        try:
            seed = int(p.stem.rsplit("_", 1)[-1])
        except ValueError:
            seed = None
        results.append({"file": f"/outputs/{p.name}", "seed": seed})
    return results


@app.get("/")
def index():
    return FileResponse("index.html")


# Serve finished sprites straight from disk. Mounted last so it doesn't shadow
# the routes above.
app.mount("/outputs", StaticFiles(directory=OUT_DIR), name="outputs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
