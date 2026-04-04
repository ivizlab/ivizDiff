# ivizDiff — Installation & Setup Guide

---

## Updating an Existing 5090 Machine (Lab)

If the lab machine already has a working ivizDiff install, you just need to pull the latest code.

Open a terminal in the project folder and run:

```bash
cd C:\_proj\iVizDiff
git pull mine main
```

If `mine` remote is not set up yet on that machine, add it first:

```bash
git remote add mine https://github.com/ivizlab/ivizDiff.git
git pull mine main
```

Then rebuild the frontend:

```bash
cd demo/realtime-img2img/frontend
npm run build
```

That's it. No Python reinstall needed — all changes are code only.

> If you get merge conflicts, the safest approach is:
> `git fetch mine && git reset --hard mine/main`
> This discards any local changes and takes the latest from GitHub.

---

## Fresh Install — RTX 3090 (24GB, Ampere)

### Prerequisites

- Windows 10/11
- NVIDIA driver 525+ (check with `nvidia-smi`)
- CUDA Toolkit 12.1 — download from nvidia.com/cuda-downloads
- Anaconda or Miniconda
- Node.js 18+ — download from nodejs.org
- Git — download from git-scm.com

### 1. Clone the repo

```bash
git clone https://github.com/ivizlab/ivizDiff.git C:\_proj\iVizDiff
cd C:\_proj\iVizDiff
```

### 2. Create conda environment

```bash
conda create -n sdiff python=3.11 -y
conda activate sdiff
```

### 3. Install PyTorch (3090 version — CUDA 12.1)

```bash
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

Verify:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
Should print `2.3.0+cu121` and `True`.

### 4. Install project dependencies

```bash
cd demo/realtime-img2img
pip install -r requirements.txt
```

### 5. Install TensorRT (optional but recommended for 3090)

TensorRT gives a significant speedup on Ampere. Follow the instructions in `docs/TENSORRT_INSTALL.md` (or skip and use `acceleration: none` in your config — see below).

### 6. Build the frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

### 7. Download models

Models are large and not in the repo. You need:

| Model | Size | Path |
|---|---|---|
| SD1.5 base (diffusers format) | ~4GB | `C:\AI\models\` |
| IPAdapter sd15 | ~300MB | `C:\AI\models\ipadapter\models\ip-adapter_sd15.bin` |
| ControlNet tile | ~1.5GB | `models/controlnet/sd15_tile/` |

Ask the lab for model files or see the shared drive.

### 8. Config for 3090

Copy the default config and set acceleration:

```bash
cp configs/NewStart.yaml configs/3090.yaml
```

Edit `configs/3090.yaml` — the key line:

```yaml
acceleration: tensorrt   # use this if TensorRT is installed
# acceleration: none     # use this if TensorRT is NOT installed
```

If using `acceleration: none`, expect **~8–12 fps** instead of 20–30fps. Still usable.

### 9. Start the system

```bash
conda activate sdiff
cd C:\_proj\iVizDiff\demo\realtime-img2img
python main.py --config ../../configs/3090.yaml
```

Open browser at `http://localhost:7860`.

---

## Day-to-Day Workflow (all machines)

### Getting the latest updates from home

```bash
cd C:\_proj\iVizDiff
git pull mine main
cd demo/realtime-img2img/frontend
npm run build
```

You only need `npm run build` if frontend files changed (`.svelte`, `.ts` files in `frontend/src/`). Python-only changes don't need a rebuild.

### Checking what changed

```bash
git log --oneline -10        # last 10 commits
git diff HEAD~1              # what changed in the last commit
```

---

## Hardware Summary

| Machine | GPU | VRAM | Expected FPS | Acceleration |
|---|---|---|---|---|
| Home / Lab | RTX 5090 | 32GB | 30+ fps | tensorrt |
| Grad | RTX 3090 | 24GB | 15–25 fps | tensorrt or none |
| Grad | GTX 1080 | 8GB | 2–5 fps | none only |

> **GTX 1080 note:** 8GB VRAM is very tight. You may need to disable IPAdapter or ControlNet in the config to fit in memory. Use `acceleration: none`. Performance will be limited — treat it as a preview/development machine only.

---

## Troubleshooting

**`torch.cuda.is_available()` returns False**
→ CUDA driver mismatch. Check `nvidia-smi` shows a GPU and reinstall the correct PyTorch build.

**TensorRT engines take forever on first run**
→ Normal. Engines compile once and cache in `engines/`. Subsequent starts are fast.

**Frontend shows blank page**
→ Run `npm run build` in `demo/realtime-img2img/frontend/` and restart.

**`ModuleNotFoundError` on startup**
→ Make sure conda env is activated: `conda activate sdiff`
