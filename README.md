# Goatface Sprite Console

A self-hosted pixel-art sprite generator — a PixelLab alternative you run on
your own hardware, built for the game **Signal Lost**. Prompt in,
palette-quantized sprite PNGs out, ready for Aseprite. No accounts, no
subscription.

- `app.py` — FastAPI backend. Detects your GPU, loads SDXL + a pixel-art LoRA,
  generates, pixelates, quantizes, saves to `outputs/`, and serves the GUI.
- `index.html` — the GUI. Plain HTML/CSS/JS, no build step.
- `requirements.txt` — Python deps (torch is installed separately, see below).

The GUI runs in your normal browser. The **backend** runs wherever the GPU is —
a rented pod for a test run, or a local card. Same files either way.

---

## 1. Test run on RunPod (RTX 3090)

The cheapest way to try this before committing to hardware. Budget a couple of
dollars for an evening; billing is per-minute while the pod runs.

**1. Create the pod** (in the browser, on runpod.io)

- Deploy → GPU Pod → pick an **RTX 3090** (24 GB is plenty for SDXL).
- Template: **RunPod PyTorch** (any recent CUDA 12.x version — torch is already
  installed in this template, so you skip the torch step below).
- Before deploying, edit the template and **expose HTTP port `8000`**.
- Deploy and wait for the pod to show **Running**.

**2. Get the code and start it** (open the pod's web terminal, paste as a block):

```bash
git clone <this-repo-url> goatface
cd goatface
pip install -r requirements.txt
python app.py
```

**3. Open the GUI**

On the pod page, click **Connect → HTTP Service [Port 8000]**. That opens the
console through RunPod's port-8000 proxy in your browser.

The status dot is amber while the model loads. **First launch downloads ~7 GB of
model weights — that is a one-time download, not training.** It's slow the first
time and instant afterwards. When the dot turns green and says *Model ready*,
generate. Each card has a save link; everything also lands in `outputs/` with
the prompt and seed embedded in the PNG.

**4. When you're done, STOP THE POD.** It bills per-minute the whole time it's
running, whether or not you're generating.

---

## 2. Local install — NVIDIA

For a machine with an NVIDIA card (~12 GB VRAM or more for SDXL; a used 3090 is
ideal).

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# torch is installed separately, matched to your CUDA version:
pip install torch --index-url https://download.pytorch.org/whl/cu121
python app.py
```

Then open <http://localhost:8000>. The header chip will read something like
`CUDA · RTX 3090 · 24GB` once the model is warm.

---

## 3. Local install — AMD / ROCm (RX 9060 XT, Linux Mint 22)

AMD cards run through ROCm. torch comes from AMD's ROCm wheel index instead of
the CUDA one:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# torch built for ROCm (check rocm.docs.amd.com for the version matching your
# driver; adjust the rocm6.x in the URL to match):
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
python app.py
```

The app detects ROCm through the same code path as CUDA — the header chip will
read `ROCm · Radeon RX 9060 XT · …`.

> ### ⚠️ Heads-up: RX 9060 XT VRAM reporting bug
> There is currently an **open ROCm bug** where this card can report only **half
> its actual VRAM** to the runtime. When it bites, SDXL either **hangs partway
> through loading** (the status dot stays amber forever) or the process
> **segfaults** as it runs out of what the driver thinks is available memory.
>
> There is no clean fix yet. The workarounds are to switch the **GPU API** dropdown
> to **CPU** (slow — minutes per image, but it completes), or to **rent a remote
> GPU** (see the RunPod section above) until the driver bug is resolved.

---

## Knobs worth knowing

- **Sprite grid** (32/64/96/128/192/256): final true resolution. Prop/environment
  sprites usually live around 64–96. **Native** skips the downscale entirely and
  quantizes the full 1024px render — palette still applies — for hero assets and
  mockups.
- **Vibrance**: when on, steers the model toward a richer, more saturated palette
  (adds saturation keywords to the prompt and pushes "muted, desaturated" onto the
  negative prompt).
- **Palette colors**: quantization limit. Lower = punchier, more "authored"
  look; 16–32 is the sweet spot.
- **Variations**: how many sprites per generate (1–8). One generation runs at a
  time to keep VRAM sane.
- **Seed**: note the seed of a keeper, then rerun it with a tweaked prompt to get
  controlled variations of the same sprite.
- **GPU API**: force Auto / CUDA-ROCm / CPU. Switching it reloads the model on the
  new device — the dot goes amber while it swaps.
- Model and LoRA are constants at the top of `app.py` (`MODEL_ID`, `LORA_ID`) —
  swapping in a different pixel-art LoRA is a one-line change.
