# CONTEXT

Canonical decision log for the Goatface Sprite Console (a self-hosted pixel-art
sprite generator for the game Signal Lost). Newest entries first. Each dated
section records the decisions made that day and the open items still in flight —
the "why" behind the code, so choices don't get re-litigated later.

---

## 2026-07-03: First live session + art direction

### DECISIONS

- **Platform proven end-to-end** on a rented RunPod RTX 3090 ($0.47/hr, billed
  per ms): deploy → `git clone` → `pip install` → `python app.py` → HTTP port
  8000. Total session cost under $1.
- **HF_TOKEN required in env** for gated models (Flux). Export it before running
  `app.py`; never commit the token.
- **Engine verdicts:**
  - *Pixel XL* — baseline is below the quality bar even with the model-card-correct
    recipe (LCM + weight 1.2, 8 steps, cfg 1.5).
  - *Flux Pixel* — runs only with `enable_model_cpu_offload` on 24GB; a single
    sample is better but still below the bar.
  - **Conclusion:** generic engines cannot hit the target aesthetic — style must
    come from a custom-trained LoRA.
- **Target aesthetic DEFINED:** high-detail "hi-bit" painterly pixel art, per Les's
  Google Flow reference image (white/purple ships, lush vibrant terrain, equipment
  visibly readable on characters). Off-the-shelf asset packs ruled out after a
  market search — this density is not sold in packs (STRANDED series et al.
  rejected as too low-detail).
- **Chosen pipeline:** Google Flow (reference image attached to every prompt + a
  written style block) generates 60–80 specimen images → curate to 40+ keepers
  (min 8 characters) in `style_set/` → train a style LoRA on a rented 3090 → add
  it as a third console engine, **"Signal Lost"**. The prompt set lives in
  `signal-lost-style-prompts.md` (add this file to the repo — Les will provide it).
- **"Hi-bit" note:** the reference style is faux-pixel (variable grid, unlimited
  palette). Console post-processing enforces a true grid + palette downstream, so
  specimens only need style-fidelity. Final crunchiness (native vs 256 vs 128,
  palette size) is a deliberate dial to be chosen later and recorded in Signal
  Lost's `ART_STYLE.md`.
- **Batch timeout issue:** the RunPod HTTP proxy drops long generations (surfaces
  as a `JSON.parse` error in the UI). Workaround: `variations=1` on slow engines.
  Proper fix queued: async job endpoint + polling.
- **Ethics rule:** never train on purchased asset packs without artist permission.
- **Character animation strategy for Signal Lost:** generate static characters →
  slice in Aseprite → Godot `Skeleton2D`/`Bone2D` puppet rigs; paper-doll layers
  for visible equipment. Frame-by-frame effects remain the only PixelLab-shaped
  gap (buy single months if ever needed).
- **Parked idea (protect this):** a quest generator for Signal Lost — a radiant
  in-game system, deterministic/NCI-shaped, no GPU. Design session pending.

### OPEN ITEMS

- Async generation (fix the proxy timeout properly)
- Upload-and-pixelate feature (see next task)
- LoRA training run procedure (Claude chat will supply when `style_set/` reaches
  40 keepers)
- Transparent-background / auto-cutout post-process for props
