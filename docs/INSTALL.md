# iVizDiff — Installation Guide

---

## Which section do you need?

| Situation | Go to |
|---|---|
| RTX 5090 / Blackwell GPU | [Section A — RTX 5090 Fresh Install](#section-a--rtx-5090-fresh-install) |
| RTX 3090 / 4090 / Ampere-Ada GPU | [Section B — RTX 3090 / 4090 Install](#section-b--rtx-3090--4090-install) |
| Already installed, pulling updates | [Updating an Existing Install](#updating-an-existing-install) |

---

## Prerequisites (all GPUs)

Install these before starting:

- **Windows 10 or 11**
- **NVIDIA driver 525+** — check with `nvidia-smi` in a terminal
- **Anaconda or Miniconda** — [miniconda.anaconda.org](https://docs.anaconda.com/miniconda/)
- **Node.js 18+** — [nodejs.org](https://nodejs.org)
- **Git** — [git-scm.com](https://git-scm.com)

---

## Section A — RTX 5090 Fresh Install

> The RTX 5090 uses the Blackwell architecture (sm_120). It requires a specific
> PyTorch build and **xformers must never be installed** — it will corrupt your environment.

### A1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/iVizDiff.git C:\_proj\iVizDiff
cd C:\_proj\iVizDiff
```

### A2. Create conda environment

```bash
conda create -n sdiff python=3.11 -y
conda activate sdiff
```

### A3. Install PyTorch (Blackwell / CUDA 12.8)

```bash
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```

Verify — this must print `2.7.0+cu128` and `True`:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

> **Warning:** Never run `pip install torch` again after this — it will overwrite the cu128 build.
> If PyTorch ever gets broken, reinstall it with the exact command above.

### A4. Install project dependencies

```bash
pip install -r requirements-5090.txt
pip install -e .
```

### A5. Download models

See the [Models section](#models) below.

### A6. Build the frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

### A7. Start

```bash
start.cmd
```

Or directly:
```bash
conda activate sdiff
cd C:\_proj\iVizDiff
python main.py
```

Open `http://localhost:7860` in your browser.

---

## Section B — RTX 3090 / 4090 Install

### B1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/iVizDiff.git C:\_proj\iVizDiff
cd C:\_proj\iVizDiff
```

### B2. Create conda environment

```bash
conda create -n sdiff python=3.11 -y
conda activate sdiff
```

### B3. Install PyTorch (CUDA 12.1)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### B4. Install project dependencies

```bash
pip install -r requirements-3090.txt
pip install -e .
```

### B5. Download models

See the [Models section](#models) below.

### B6. Build the frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

### B7. Start

```bash
start.cmd
```

Open `http://localhost:7860` in your browser.

---

## Models

Models are large files and are not included in the repo. You need to download them separately and place them in the correct folders.

### Model folder structure

```
iVizDiff/
├── models/
│   ├── sd/
│   │   └── PerfectPhotonV2.1/     ← or any SD1.5 diffusers model
│   ├── ipadapter/
│   │   └── models/
│   │       ├── ip-adapter_sd15.bin
│   │       └── image_encoder/     ← full folder from HuggingFace
│   └── controlnet/
│       └── sd15_tile/             ← full folder from HuggingFace
└── engines/                       ← TensorRT engines, compiled automatically on first run
```

---

### 1. Base diffusion model (SD1.5)

iVizDiff requires an **SD1.5 model in diffusers format**.

> **NOT compatible with SDXL, SD2, or SD3 models.**

**Option A — Free model (downloads automatically, no files needed):**

Edit `configs/NewStart.yaml` and change `model_id` to:
```yaml
model_id: "runwayml/stable-diffusion-v1-5"
```
This downloads ~4GB from HuggingFace on first run.

**Option B — PerfectPhotonV2 (the lab default, better quality):**

1. Download from CivitAI: search "PerfectPhoton V2" on [civitai.com](https://civitai.com)
2. The CivitAI download is a `.safetensors` file — convert it to diffusers format:
   ```bash
   python -c "
   from diffusers import StableDiffusionPipeline
   pipe = StableDiffusionPipeline.from_single_file('path/to/PerfectPhotonV2.1.safetensors')
   pipe.save_pretrained('models/sd/PerfectPhotonV2.1')
   "
   ```
3. The `configs/NewStart.yaml` already points to `models/sd/PerfectPhotonV2.1` — no changes needed.

Any other SD1.5 diffusers-format model placed in `models/sd/` will work — just update `model_id` in your config.

---

### 2. IPAdapter

1. Go to [huggingface.co/h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter)
2. Download:
   - `models/ip-adapter_sd15.bin` → place at `models/ipadapter/models/ip-adapter_sd15.bin`
   - `models/image_encoder/` (the whole folder) → place at `models/ipadapter/models/image_encoder/`

The `configs/NewStart.yaml` already points to these paths — no changes needed.

---

### 3. ControlNet (sd15_tile)

1. Go to [huggingface.co/lllyasviel/control_v11f1e_sd15_tile](https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile)
2. Download the full repository (all files) into `models/controlnet/sd15_tile/`

**Or** let it download automatically — edit `configs/NewStart.yaml` and change the controlnet `model_id` to:
```yaml
model_id: "lllyasviel/control_v11f1e_sd15_tile"
```
This downloads on first run and caches in HuggingFace's local cache.

---

### 4. TensorRT engines

Engines are compiled automatically on first run when `acceleration: tensorrt` is set.
This takes **15–30 minutes** but only happens once per resolution/model combination.
Compiled engines are stored in `engines/` and reused on all subsequent starts.

If you want to skip TensorRT for now, use `acceleration: none` in your config — slower but works immediately.

---

## Updating an Existing Install

```bash
cd C:\_proj\iVizDiff
git pull origin main
cd frontend && npm run build && cd ..
```

Python dependencies rarely change. If you see import errors after an update:
```bash
pip install -r requirements-5090.txt   # or requirements-3090.txt
pip install -e .
```

---

## Troubleshooting

**`torch.cuda.is_available()` returns `False`**
→ CUDA driver mismatch. Check `nvidia-smi` shows your GPU. Reinstall the correct PyTorch build for your CUDA version.

**`ModuleNotFoundError: No module named 'streamdiffusion'`**
→ Run `pip install -e .` from the repo root.

**TensorRT engine build takes a long time on first start**
→ Normal. Engines compile once and cache in `engines/`. Subsequent starts are fast.

**Frontend shows blank page**
→ Run `npm run build` in `frontend/` and restart.

**RTX 5090: `xformers` error or PyTorch version changed**
→ Reinstall PyTorch cu128:
```bash
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```

**Out of memory on startup**
→ Disable IPAdapter or ControlNet in `configs/NewStart.yaml` by setting `enabled: false`. Use `acceleration: none` on GPUs with less than 16GB VRAM.
