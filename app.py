"""
Goatface Sprite Console — backend
=================================
One file. It does four jobs:
  1. Detects the GPU backend (ROCm / CUDA / MPS / CPU) at startup.
  2. Loads a pixel-art image model (SDXL + pixel-art LoRA) in the background.
  3. Exposes POST /generate — takes a prompt, returns finished sprite PNGs.
  4. Serves index.html (the GUI) at /.

Post-processing baked in: every image is downscaled to a true pixel grid
(NEAREST, so no blur) and quantized to a limited palette — the two steps
that make diffusion output look like real pixel art instead of a photo
of pixel art.

Run it:   python app.py
Open:     http://localhost:8000   (or the pod's proxy URL on RunPod)
"""

import time
import threading
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# ---------------------------------------------------------------- settings

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_ID = "nerijs/pixel-art-xl"      # pixel-art fine-tune, trigger word: "pixel"
TRIGGER = "pixel art"                # prepended to every prompt
NEGATIVE = "blurry, photo, realistic, 3d render, jpeg artifacts, watermark, text"

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
# show "warming up" instead of hanging. First run downloads ~7 GB of weights.

def load_model(prefer="auto"):
    # Flip back to loading so a /backend swap shows the warming state in the UI.
    state["status"] = "loading"
    state["error"] = None
    state["pipe"] = None

    try:
        detect_backend(prefer)
        device = state["device"]
        on_gpu = device in ("cuda", "mps")
        # fp16 weights on GPU keep VRAM down; CPU needs fp32 for correctness.
        dtype = torch.float16 if on_gpu else torch.float32

        from diffusers import StableDiffusionXLPipeline

        if on_gpu:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                MODEL_ID, torch_dtype=dtype, variant="fp16", use_safetensors=True,
            )
        else:
            # No fp16 variant on CPU — load the full-precision weights.
            pipe = StableDiffusionXLPipeline.from_pretrained(
                MODEL_ID, torch_dtype=dtype, use_safetensors=True,
            )

        pipe.load_lora_weights(LORA_ID)
        pipe.to(device)

        # Small VRAM saver, costs almost nothing in speed. GPU-only — it's a
        # memory trick and CPU isn't memory-bound the same way.
        if on_gpu:
            pipe.enable_vae_tiling()

        state["pipe"] = pipe
        state["status"] = "ready"
    except Exception as e:  # surface load failures to the GUI
        state["status"] = "error"
        state["error"] = str(e)


@app.on_event("startup")
def startup():
    """Kick off the first load in the background so the GUI is instant."""
    threading.Thread(target=load_model, args=("auto",), daemon=True).start()


# ---------------------------------------------------------------- pixel post

def pixelate(img: Image.Image, grid: int, colors: int) -> Image.Image:
    """Turn raw 1024x1024 diffusion output into honest pixel art:
    shrink to the target grid (this *is* the pixelation), then
    quantize to a limited palette like a real sprite sheet."""
    small = img.resize((grid, grid), Image.NEAREST)
    small = small.quantize(colors=colors, method=Image.MEDIANCUT).convert("RGB")
    return small


# ---------------------------------------------------------------- api

class GenRequest(BaseModel):
    prompt: str
    grid: int = 96          # 32 / 64 / 96 / 128 — final sprite resolution
    colors: int = 24        # palette size after quantization
    count: int = 4          # how many variations
    seed: int | None = None # set for reproducible results


class BackendRequest(BaseModel):
    device: str = "auto"    # "auto" | "cuda" | "cpu"


@app.get("/health")
def health():
    return {
        "status": state["status"],
        "error": state["error"],
        "backend": state["backend"],
        "device_name": state["device_name"],
        "vram_gb": state["vram_gb"],
    }


@app.post("/backend")
def set_backend(req: BackendRequest):
    """Reload the pipeline on a different device.

    Returns immediately; the swap happens on a background thread and the status
    goes back to "loading" while it warms up.
    """
    device = req.device if req.device in ("auto", "cuda", "cpu") else "auto"
    threading.Thread(target=load_model, args=(device,), daemon=True).start()
    return {"status": "loading", "device": device}


@app.post("/generate")
def generate(req: GenRequest):
    if state["status"] != "ready" or state["pipe"] is None:
        return JSONResponse({"error": f"model not ready ({state['status']})"}, 503)

    # Clamp everything so a bad request can't ask for a 900-color, 50-image job.
    req.count = max(1, min(req.count, 8))
    req.grid = req.grid if req.grid in (32, 64, 96, 128) else 96
    req.colors = max(4, min(req.colors, 64))

    device = state["device"]
    results = []
    with gen_lock:
        pipe = state["pipe"]
        for i in range(req.count):
            # Each variation gets its own seed. A pinned seed steps per image so
            # the batch is reproducible but not four identical sprites.
            seed = (req.seed + i) if req.seed is not None else torch.seed() % (2**31)
            g = torch.Generator(device=device).manual_seed(seed)

            image = pipe(
                prompt=f"{TRIGGER}, {req.prompt}",
                negative_prompt=NEGATIVE,
                num_inference_steps=28,
                guidance_scale=7.0,
                width=1024, height=1024,
                generator=g,
            ).images[0]

            sprite = pixelate(image, req.grid, req.colors)

            name = f"{int(time.time())}_{seed}.png"
            path = OUT_DIR / name
            # Stash the prompt + seed inside the PNG so files stay self-describing.
            meta = PngInfo()
            meta.add_text("prompt", f"{TRIGGER}, {req.prompt}")
            meta.add_text("seed", str(seed))
            sprite.save(path, pnginfo=meta)

            results.append({"file": f"/outputs/{name}", "seed": seed})

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
