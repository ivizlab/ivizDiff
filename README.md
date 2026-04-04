# iVizDiff

Real-time AI video transformation for live performance and creative research.

A webcam or video file is fed into a Stable Diffusion pipeline and the transformed output streams back to a browser at 15–30 fps. Style images, ControlNet, IPAdapter, and live input controls (microphone, breath, gamepad) let you shape the output in real time. From ivizlab.org at SFU and forked from @ryanontheinside and @livepeer

---

## Hardware requirements

| GPU | CUDA | Notes |
|---|---|---|
| RTX 5090 / 5080 (Blackwell) | 12.8 | Follow the **5090 install path** in `docs/INSTALL.md` |
| Any other NVIDIA GPU (3080, 3090, 4080, 4090, laptop GPUs) | 12.1–12.4 | Follow the **standard install path** in `docs/INSTALL.md` |

Other GPUs with 8GB+ VRAM may work on the standard path but are untested.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/iVizDiff.git
cd iVizDiff

# 2. Create conda environment
conda create -n sdiff python=3.11
conda activate sdiff

# 3. Install PyTorch and dependencies — choose your GPU
#    RTX 5090:  see docs/INSTALL.md Section A
#    All other GPUs:  see docs/INSTALL.md Section B

# 4. Install the iVizDiff library
pip install -e .

# 5. Build the frontend
cd frontend && npm install && npm run build && cd ..

# 6. Start
start.cmd          # Windows
# or
python main.py     # directly
```

Open `http://localhost:7860` in your browser.

---

## Documentation

| Doc | Contents |
|---|---|
| [`docs/INSTALL.md`](docs/INSTALL.md) | Full install guide, GPU-specific steps, model downloads |
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | UI walkthrough, parameters, YAML config |
| [`docs/FEATURES.md`](docs/FEATURES.md) | Recording output video, input calibration, snapshot save/load |

---

## Project structure

```
iVizDiff/
├── main.py                  backend — FastAPI server, pipeline, WebSocket stream
├── img2img.py               pipeline wrapper
├── streamdiffusion/         SD pipeline library (modified from StreamDiffusion)
├── frontend/                SvelteKit UI
├── configs/
│   ├── NewStart.yaml        default configuration — start here
│   └── _examples/           reference configs for ControlNet, IPAdapter etc.
├── images/inputs/           style images for IPAdapter
├── snapshots/               saved PNG snapshots with embedded params
├── videos/                  recorded output videos
└── docs/                    install and usage guides
```

---

## Key parameters

| Parameter | What it does |
|---|---|
| **IPAdapter scale** | How strongly the style image shapes the output |
| **ControlNet strength** | How closely output follows input structure |
| **Timestep Step 1 / Step 2** | Range 0–49 — higher = more creative, lower = closer to input |
| **Guidance scale** | CFG strength — keep low (1.0–2.0) for real-time use |
| **Prompt blend** | Crossfade between two prompts in real time |

---


## License

Apache 2.0


- ivizlab.org - 