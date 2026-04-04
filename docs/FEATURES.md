# Img2Img Example

[English](./README.md) | [日本語](./README-ja.md)

<p align="center">
  <img src="../../assets/img2img1.gif" width=80%>
</p>

<p align="center">
  <img src="../../assets/img2img2.gif" width=80%>
</p>


This example, based on this [MPJEG server](https://github.com/radames/Real-Time-Latent-Consistency-Model/), runs image-to-image with a live webcam feed or screen capture on a web browser.

## Features

- **Standard Mode**: Basic image-to-image generation with SD-Turbo
- **ControlNet Mode**: Enhanced generation with ControlNet support (depth, canny, pose, etc.)
- **Real-time streaming**: WebRTC/WebSocket based streaming for low latency
- **Web interface**: No desktop app required, runs in browser

## Usage
You need Node.js 18+ and Python 3.10 to run this example.
Please make sure you've installed all dependencies according to the [installation instructions](../../README.md#installation).

### Standard Mode (Default)
```bash
cd frontend
npm i
npm run build
cd ..
pip install -r requirements.txt
python main.py --acceleration tensorrt   
```

or 

### Quick Start Script
```
chmod +x start.sh
./start.sh
```

Then open `http://0.0.0.0:7860` in your browser.
(*If `http://0.0.0:7860` does not work well, try `http://localhost:7860`)

## ControlNet Configuration

When using ControlNet mode, you can specify:
- **Model**: Base diffusion model (SD1.5, SD-Turbo, SDXL-Turbo)
- **ControlNets**: One or multiple ControlNet models with preprocessors
- **Parameters**: Generation settings, temporal consistency, acceleration options

See the [ControlNet configuration examples](../../configs/controlnet_examples/) for detailed YAML configuration options.

### ControlNet Mode
To use ControlNet, provide a YAML configuration file:

```bash
python main.py --acceleration tensorrt --controlnet-config /path/to/config.yaml
```

### Running with Docker

```bash
docker build -t img2img .
docker run -ti -e ENGINE_DIR=/data -e HF_HOME=/data -v ~/.cache/huggingface:/data  -p 7860:7860 --gpus all img2img
```

Where `ENGINE_DIR` and `HF_HOME` set a local cache directory, making it faster to restart the docker container.

## Command Line Options

```
--host HOST                    Host address (default: 0.0.0.0)
--port PORT                    Port number (default: 7860)
--controlnet-config PATH       Path to ControlNet YAML configuration (optional)
--acceleration ACCEL           Acceleration type: none, xformers, sfast, tensorrt
--taesd / --no-taesd          Use Tiny Autoencoder (default: enabled)
--engine-dir DIR              TensorRT engine directory
--debug                       Enable debug mode
```

---

## ivizLab Features Guide

This section covers features added for live performance and teaching use.

---

### Recording Output Video

You can record the generated video output directly from the browser — no screen capture software or OBS needed. The recording reads directly from the diffusion output stream, so it has **no impact on FPS**.

#### How to record

1. Start the stream (click **Start Stream**).
2. In the bottom-right corner of the output video, click the **record button** (red circle inside a ring).
3. The button turns into a red **stop square** and a pulsing **REC** badge appears in the top-left corner.
4. When you are done, click the **stop square**.
5. A **Save As** dialog opens — choose a location and filename. The file is saved as `.webm` (plays in any modern browser, VLC, or can be converted with ffmpeg).

#### Resolution tip

The recording captures the exact pixel dimensions that the pipeline is outputting. If you want higher-resolution recordings:

- Go to the **Upscale** control and set it to **2x** (or 4x) **before** clicking record.
- At 2x, a 512×512 pipeline outputs 1024×1024 to the stream, and that is what gets recorded.
- Changing the upscale setting mid-recording is not recommended — set it first, then record.

#### File format

Files are saved as **WebM / VP9**. To convert to MP4:
```
ffmpeg -i recording_20260403_143000.webm -c:v libx264 output.mp4
```

---

### Microphone and Breath Input — Noise Floor Calibration

The **Input Controls** panel lets you map a microphone volume or breath signal to any pipeline parameter (IPAdapter scale, ControlNet strength, prompt blend weight, etc.). In a noisy room the ambient noise can prevent the value from ever reaching its minimum, making fine control difficult.

The calibration tools let you teach the system what "silence" and "full signal" look like for your specific room and microphone.

#### Calibration controls

Each microphone or breath input control has a calibration row showing:

```
[min – max]   Noise Floor 3s   Full Range 5s   Reset
```

The `[min – max]` display shows the current calibrated range. Raw input is remapped so that `min` maps to 0 and `max` maps to 1 before sensitivity and parameter scaling are applied.

#### Step 1 — Set the noise floor (most important in loud rooms)

1. Start the input control (the control must be **active/running**).
2. Make sure the room is as quiet as it will be during your performance — **don't speak or breathe heavily**.
3. Click **Noise Floor 3s**.
4. The hint text changes to *"Stay quiet — sampling room noise floor…"*
5. After 3 seconds the `min` value updates to the peak noise level observed, plus a small safety margin.
6. If the reading was too high (someone made noise during the sample), just click **Noise Floor 3s** again — it cancels the previous run and starts fresh.

#### Step 2 — Set the full range (optional but recommended)

1. Click **Full Range 5s**.
2. The hint text changes to *"Recording — perform your full input range now…"*
3. During the 5 seconds: start at silence, build to your maximum breath/volume, then return to silence. This captures both the floor and ceiling.
4. After 5 seconds the `[min – max]` range updates.

> **Tip:** If you already set the noise floor in Step 1, the Full Range pass will overwrite `min` with the true quiet value observed during the 5-second window. Running **Noise Floor 3s** first then **Full Range 5s** is the most precise workflow.

#### Resetting calibration

Click **Reset** to return to the default `[0.000 – 1.000]` full range, removing all calibration.

#### Typical workflow for a noisy room

```
1. Open Input Controls → add a Microphone control
2. Assign it to a parameter (e.g. IPAdapter Scale, range 0.0 – 1.0)
3. Click Start on the control
4. Click Noise Floor 3s → stay quiet → wait for it to finish
5. Check the [min – max] display — min should now reflect the room noise
6. Perform normally — the mapped parameter will now reach 0 when you are quiet
   and 1 at your loudest breath/voice
7. If the range still feels off, adjust the Sensitivity slider or re-run calibration
```
