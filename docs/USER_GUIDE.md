# iVizDiff — User Guide

A real-time AI video system that transforms live webcam or video input into moving artwork using Stable Diffusion. Every output frame is shaped by four simultaneous inputs working together: your **text prompts** steer the visual style; an **IPAdapter style image** pulls the output toward a particular aesthetic; the **live camera or video** provides the structural and compositional anchor; and a set of **parameters** governs how faithfully the output tracks the camera, how stable and smooth the stream is over time, and how aggressively the model transforms the input. The result is a live, continuously generated video blending all four influences in real time, typically at 30+ frames per second.

---

## Starting Up

Double-click **`StartiVizDiff.cmd`** in `C:\_proj\iVizDiff`.

The backend starts and your browser opens automatically at `http://localhost:7860`. Wait until the UI is fully loaded — you will see control panels on left and right and the output area in the centre.

Click **Start Stream** in the top bar. The system loads the default configuration and begins generating.

> The first run after a system restart will be slower while TensorRT engines compile. Subsequent runs start much faster.

---

## Choosing Your Input

Under **Video Input** on the left panel you choose what feeds into the pipeline.

### Webcam (default)
Select your webcam from the **Video Input** dropdown. Whatever is in frame becomes the input — the AI transforms it in real time.

### Video File
Place your video file in **`videos/input/`** (inside the iVizDiff folder). Then select it from the Video Input dropdown. The video loops continuously. You can switch the source before or after starting the stream — live switching is supported.

### Resolution
Below the input preview are resolution controls. Lower resolution = faster generation (higher FPS). Higher resolution = more detail but slower.

---

## Creating a Look — Prompts

On the left panel are **two prompt fields** (Prompt 1 and Prompt 2).

- Each prompt describes the visual style — colours, textures, atmosphere, artistic style, lighting.
- The prompt does **not** describe what is in the scene. The scene comes from your camera. The prompt describes *how it looks*.
- The **blend slider** between them interpolates. You can sit at one end, the other, or anywhere in between for a mixture of both styles.
- Slowly moving the blend slider over time creates a smooth morph between two looks.

**Tips:**
- Start with one prompt and get a look you like, then write a second and blend.
- Style words work well: `"oil painting, warm light, impressionist"` / `"neon, dark, cyberpunk"` / `"soft watercolour, pastel"`.

---

## Loading a Style Image — IPAdapter

The **IPAdapter** section (right panel, bottom) lets you load a reference image that pulls the output toward a particular visual aesthetic.

- Click **Upload** to load a style image from `images/input/` (or anywhere).
- The **IPAdapter Scale** slider controls how strongly it influences the output.
  - **0.3** — subtle style influence, prompt and webcam dominate
  - **0.7** — strong style pull, gives the output its illustrated or painterly character
  - **1.0** — everything looks like the reference image regardless of camera
- You can load a second style image and blend between the two.

This is one of the most direct controls for the overall "look" — a painting, texture, or photograph loaded here will push the output toward that aesthetic even without any prompt describing it.

---

## Saving Your Work — Snapshots

Click the **snapshot button** (camera icon in the output area) to save the current frame as a PNG.

Snapshots go to **`snapshots/`** with all current settings embedded — prompts, IPAdapter scale, ControlNet strength, timesteps, guidance, everything. Loading a snapshot later restores all those settings exactly.

---

## Right Panel — Parameters

These control the internal diffusion process. Hover over any label for a tooltip.

---

### ControlNet

#### Strength (default 0.20)
Controls how tightly the output tracks the **structure of the camera input** — your pose, the geometry of the scene, edges and shapes. At 0.20 it is gentle, preserving basic structure without overriding the style. Higher values make the output hug the source geometry more tightly; too high and it fights the creative prompts.

#### Feedback Strength (default 0.20)
This is unique to this setup. Instead of a traditional preprocessor (like Canny edges), the **feedback** preprocessor blends the current webcam frame with the **previous output frame**.

- **0.0** — pure webcam input, no memory
- **0.20** — mostly camera with a subtle temporal memory, keeps the output stable rather than flickering
- **1.0** — only the last generated frame feeds back in, creating a looping/dreaming effect disconnected from the camera

This is what gives the stream its temporal stability. Higher values create a more self-referential, evolving quality; the image builds on itself rather than following the camera moment to moment.

---

### Timesteps — Step 1 and Step 2 (defaults: 32 and 45)

These are indices into the 50-step diffusion schedule. StreamDiffusion does not run all 50 steps per frame — it samples at just these two points. Think of them as where in the denoising process each frame starts and stops.

- **Lower indices** (earlier in diffusion) = more creative, looser, more abstract
- **Higher indices** (later in diffusion) = more refined, structured, photorealistic
- The **gap between them** affects how much transformation happens per frame

| Mood | Step 1 | Step 2 |
|---|---|---|
| Default balanced | 32 | 45 |
| More abstract / dreamlike | 20–25 | 45 |
| More literal / structured | 38–42 | 47 |

Moving both lower makes the output more hallucinatory. Moving both higher makes it more photorealistic. If you change **Inference Steps** (the schedule size), revisit these values too.

---

### Streaming Parameters

#### Guidance Scale (default 1.10)
How strongly the **text prompt** steers the image. Standard SD uses values like 7–12, but StreamDiffusion with LCM works best at very low values (1.0–2.0). At 1.10 the prompt is a gentle nudge — the model follows it loosely, which is appropriate for smooth real-time streaming. Higher values make the output more prompt-obedient but can cause flickering or artifacts at streaming speeds.

#### Delta (default 0.70)
A StreamDiffusion-specific parameter — a noise multiplier that maintains variation between frames and prevents the output from "locking up" into a static image. Too low and the stream can freeze or loop. Too high and it becomes unstable. Leave at 0.70 unless experimenting.

#### Inference Steps (default 50)
The total size of the denoising schedule that the timestep indices are drawn from — not the number of steps actually run per frame. Changing this reshapes the whole schedule curve, so if you change it you will want to revisit your Step 1 and Step 2 values. Leave at 50 in normal use.

---

### IPAdapter Scale (right panel slider)
Same as the scale in the IPAdapter image upload section — duplicated here for convenience. See above for full explanation.

---

## How the Three Main Controls Interact

IPAdapter scale, ControlNet strength, and Guidance scale all compete with each other. No single one is fully independent.

| Goal | What to do |
|---|---|
| "More me in the output" | Raise ControlNet strength, lower IPAdapter scale |
| "More style image in the output" | Raise IPAdapter scale, lower ControlNet strength |
| "More prompt-driven output" | Raise Guidance scale, lower IPAdapter scale |
| "Looser, more dreamlike" | Lower Step 1 (toward 20–25), lower ControlNet strength |
| "More stable, less flickery" | Raise Feedback strength slightly (0.25–0.35) |

The timestep indices act as a global creativity/fidelity dial underneath all of this — they determine how far into the denoising process each frame is processed, which affects how much any of the other inputs can actually shape the result.

---

## Left Panel — Input Controls

Below the prompt section is the **Input Controls** panel. This maps a physical input to any parameter slider, so you can drive guidance, IPAdapter scale, ControlNet strength, or anything else with your body, breath, or a controller — in real time, without touching the UI.

### Microphone
Maps the **volume level** of ambient sound or your voice to a parameter. Loud = high value, quiet = low value. Reactive and instant — good for music or voice directly driving a visual effect.

### Gamepad Controller
Maps a gamepad axis or trigger to any parameter. Connect a USB controller and assign axes to whichever sliders you want to control live.

### Breath Controller
Tracks the **rhythm of your breathing** specifically — ignoring speech and room noise. Gives a slow, organic wave that follows your breath cycle (inhale → exhale → inhale, roughly 4–8 seconds per cycle).

Two signals:
- **Amplitude** — rises on exhale, falls on inhale. Smooth and body-paced. Good for IPAdapter scale, ControlNet strength — parameters you want to pulse organically.
- **BPM** — your breathing rate over recent cycles, normalised. Changes very slowly. Slow breathing = calmer image character; fast breathing = more energetic. Good for long-arc character shifts.

**Mic vs Breath in practice:** Mic reacts to what is happening in the room right now. Breath follows your body's internal rhythm regardless of room sound. In a performance or installation, a viewer standing near the mic and simply breathing naturally will subtly animate the image — no gestures, no UI, no conscious control. The image breathes with them.

#### Starting the Breath Controller

The breath controller is a **separate process** that runs in its own terminal. Start it before or after iVizDiff — they find each other automatically.

Open a terminal and run:
```
cd C:\_proj\breath
python breath.py --ws
```

You will see a live ASCII breath meter and JSON events printed as you breathe. Leave this window running.

Then in iVizDiff:
1. Expand **Input Controls** on the left panel
2. Click **Add Breath Control**
3. Set: **Signal** (Amplitude or BPM), **Parameter** to drive, **Min / Max** range, **Sensitivity**, **Update Rate**
4. Click **Start**

Status badges show connection state:

| Badge | Meaning |
|---|---|
| **DISCONNECTED** (red) | breath.py is not running or not found yet |
| **WAITING** (amber) | Connected but no breath events received yet |
| **LIVE** (green) | Events flowing, parameter is being driven |
| **↑ inhale / ↓ exhale** | Current breath phase |

**Calibration:** The amplitude signal uses a rolling maximum normaliser — it learns your breath range over the first 5–10 breath cycles. It will under-range at first. This is normal. After calibration it stabilises. If it never reaches your max value, raise Sensitivity slightly (try 1.2–1.5), or lower the Max value to match your actual peak.

---

## Folder Layout

| Folder | Purpose |
|---|---|
| `videos/input/` | Video files to use as input source |
| `images/input/` | Style reference images for IPAdapter |
| `snapshots/` | Saved output frames (PNG with all settings embedded) |
| `configs/` | YAML config files — save and load full parameter sets |
| `engines/` | TensorRT compiled engines (generated automatically, do not delete) |

---

## Quick Reference

| Control | What it does | Default | Range |
|---|---|---|---|
| **Prompt 1 / 2** | Visual style description | — | — |
| **Prompt blend** | Interpolate between two prompts | — | 0.0–1.0 |
| **IPAdapter Scale** | Style image influence strength | 0.70 | 0.0–1.0 |
| **ControlNet Strength** | How tightly output tracks camera structure | 0.20 | 0.0–1.0 |
| **Feedback Strength** | How much previous output frame feeds back in | 0.20 | 0.0–1.0 |
| **Step 1** | Start of diffusion pass (lower = more creative) | 32 | 0–49 |
| **Step 2** | End of diffusion pass | 45 | 0–49 |
| **Guidance Scale** | Prompt adherence (keep low for streaming) | 1.10 | 1.0–3.0 |
| **Delta** | Noise variation between frames | 0.70 | — |
| **Inference Steps** | Schedule size (leave at 50) | 50 | — |
