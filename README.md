# iVizDiff

Real-time AI video transformation for live performance and creative research — built on parts of **[StreamDiffusion](https://github.com/daydreamlive/StreamDiffusion)**, a real-time diffusion pipeline that runs Stable Diffusion 1.5 in img2img mode to continuously reshape a live webcam or video feed into flowing AI-generated video.

Every output frame is shaped by four inputs working together: the **live camera or video** provides the structural anchor — pose, layout, composition; **text prompts** steer style and content; **style image(s) (IPAdapter)** pull the look toward a chosen aesthetic; and a set of **parameters** (ControlNet strength, feedback, guidance scale, timesteps) govern how closely the output tracks the input and how aggressively it's transformed. Any of these can also be driven live by a physical input — hand gesture, microphone, breath, or gamepad. The result streams back to a browser at 15–30 fps. The system can generate images or recorded video and has several features for artistic journeying, including saving/loading any part of your journey.

From ivizlab.org at SFU (working with and inspired by work from @ryanontheinside and @livepeer.)


## Hardware requirements

| GPU | CUDA | Notes |
| - | - | - |
| RTX 5090 / 5080 (Blackwell) | 12.8 | Follow the **5090 install path** in `docs/INSTALL.md` |
| Any other NVIDIA GPU (3080, 3090, 4080, 4090, laptop GPUs) | 12.1–12.4 | Follow the **standard install path** in `docs/INSTALL.md` |

Other GPUs with 8GB+ VRAM may work on the standard path but are untested.


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

# 5. Download required models (base SD1.5 checkpoint, IPAdapter, ControlNet)
#    and place them under models/ — the app will not start without them.
#    See the "Models" section in docs/INSTALL.md for exact download links
#    and folder locations.

# 6. Build the frontend
cd frontend && npm install && npm run build && cd ..

# 7. Start
start.cmd          # Windows
# or
python main.py     # directly
```

Open `http://localhost:7860` in your browser.


## Documentation

| Doc | Contents |
| - | - |
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | Full UI walkthrough — every panel and parameter, YAML config, LoRAs, recording, snapshots, hand/mic/gamepad/breath input mapping |
| [`docs/INSTALL.md`](docs/INSTALL.md) | Full install guide, GPU-specific steps, model downloads |


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


## Key parameters

| Parameter | What it does |
| - | - |
| **IPAdapter scale** | How strongly the style image shapes the output |
| **ControlNet strength** | How closely output follows input structure |
| **Timestep Step 1 / Step 2** | Range 0–49 — higher = more creative, lower = closer to input |
| **Guidance scale** | CFG strength — keep low (1.0–2.0) for real-time use |
| **Prompt blend** | Crossfade between two prompts in real time |


## License

Apache 2.0

- ivizlab.org -
