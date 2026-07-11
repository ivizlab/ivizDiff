# iVizDiff — User Guide

**iVizDiff is built on StreamDiffusion, a real-time image generation pipeline that runs Stable Diffusion 1.5 in img2img mode** — continuously reshaping a live webcam or video feed into flowing AI-generated video, rather than generating images from scratch. Every output frame is shaped by four inputs working together: your **text prompts** steer style and content; an **IPAdapter style image** pulls the look toward a chosen aesthetic; the **live camera or video** provides the structural anchor — pose, layout, composition; and a set of **parameters** (ControlNet strength, feedback loop, guidance scale, delta, timestep indices) govern how closely the output tracks the camera, how stable it stays over time, and how aggressively it's transformed. Any of these parameters can also be driven live by a physical input — hand gesture read from video, a game controller, microphone volume, or breath — each remappable to whatever range you need. Together this produces a continuous, real-time stream, typically 30+ fps on a modern GPU. *(See Technical Details at the end of this guide for how StreamDiffusion actually achieves that speed.)*

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
Place your video file in **`videos/input/`** (inside the iVizDiff folder). Then select it from the Video Input dropdown. The video always loops continuously. You can switch the source before or after starting the stream — live switching is supported.

**Speed** — 1×–5× buttons above the source dropdown control playback speed of the video file (does not apply to webcam). Higher values play the source faster, which changes how much motion the model sees frame to frame.

### Resolution
Below the input preview are resolution controls. Lower resolution = faster generation (higher FPS). Higher resolution = more detail but slower.

---

## Creating a Look — Prompts

On the left panel are **prompt fields**, starting with Prompt 1 and Prompt 2.

- Each prompt describes the visual style — colours, textures, atmosphere, artistic style, lighting.
- The prompt does **not** describe what is in the scene. The scene comes from your camera. The prompt describes *how it looks*.
- The **blend slider** between Prompt 1 and Prompt 2 interpolates. You can sit at one end, the other, or anywhere in between for a mixture of both styles.
- Slowly moving the blend slider over time creates a smooth morph between two looks.

**More than two prompts:** Click **+ Add Prompt** to add additional prompts beyond the first two, each with its own weight.

- **Normalize Prompt Weights** (checkbox) — when on, all prompt weights are normalized to sum to 1. When off, weights above 1 amplify that prompt's influence rather than just shifting the mix.
- **Interpolation Method** — how multiple prompt embeddings are combined. Default is **SLERP (Spherical Linear)**.

**Tips:**
- Start with one prompt and get a look you like, then write a second and blend.
- Style words work well: `"oil painting, warm light, impressionist"` / `"neon, dark, cyberpunk"` / `"soft watercolour, pastel"`.

---

## Loading a Style Image — IPAdapter

The **IPAdapter** section (right panel, top) lets you load one or two reference images that pull the output toward a particular visual aesthetic.

- Click **Upload Style Image** to load your first style image (Style A) from `images/input/` (or anywhere).
- Click **Replace B** to load a second style image (Style B) alongside the first.
- With two images loaded, the **A → B Blend** slider (0.00–1.00) interpolates between the two styles — 0.00 is pure Style A, 1.00 is pure Style B, and anywhere between mixes both. This is separate from the Scale slider below.
- The **Scale** slider controls overall IPAdapter strength, regardless of whether you're using one or two style images:
  - **0.3** — subtle style influence, prompt and webcam dominate
  - **0.7** — strong style pull, gives the output its illustrated or painterly character
  - **1.0** — everything looks like the reference image(s) regardless of camera

This is one of the most direct controls for the overall "look" — a painting, texture, or photograph loaded here will push the output toward that aesthetic even without any prompt describing it. Because it's the topmost section on the right panel, it sits right next to ControlNet — the two controls you'll reach for together most often.

---

## Using LoRAs

LoRAs let you push the output toward a specific trained style — your own art style, a particular character look, a texture — stacked on top of everything else the pipeline is doing.

### LoRA Panel

The **LoRA panel** (right panel, above Output Upscale) manages which LoRAs are active:

- A dropdown lists every LoRA in your LoRA folder. Pick one and click **Add**.
- Click **Apply & Restart** to actually load it into the pipeline — the stream restarts and comes back in a few seconds. Adding or removing a LoRA requires this restart; it isn't live.
- Once a LoRA is loaded and active, it gets its own **weight slider** (0.0–1.0) that you can adjust live, with no restart needed — this is the per-LoRA scale.
- If nothing is loaded, the panel shows **"No LoRAs active."**

Multiple LoRAs can be added and stacked at once, each with its own weight slider.

### LoRA Config (YAML)

LoRAs can also be predefined in your config file under a `loras:` section, in the same pattern as `ipadapters` and `controlnets`:

```yaml
loras:
  - path: "D:/_proj/__artLora/sd15"
    weight_name: "yourlora-000007.safetensors"
    adapter_name: "yourstyle"
    trigger_word: "yourstyle_trigger"
    scale: 0.7
    enabled: true
```

- **`trigger_word`** (optional) — when the LoRA is enabled, its trigger word is automatically prepended to the prompt; when disabled, it's automatically removed. You don't need to manage this by hand in the prompt fields.
- **`scale`** sets the starting weight; it can still be adjusted live from the LoRA panel's weight slider afterward.

**Note:** LoRAs must be loaded before TensorRT engines are built. Adding a new LoRA (via the dropdown + Add + Apply & Restart, or by editing the config) requires a stream restart to take effect — only the weight slider on an already-loaded LoRA is live.

---

## Output Controls

Below the output preview is a row of five icons, left to right:

1. **Record** (red circle) — starts/stops recording the output stream to video.
2. **Fullscreen** (expand icon) — expands the output to fill the screen.
3. **Save Snapshot** (camera icon) — saves the current frame as a PNG with all current settings embedded: prompts, IPAdapter scale, ControlNet strength, timesteps, guidance, LoRA weights, everything.
4. **Load Parameters** (download icon) — loads a previously saved snapshot's embedded settings back into the UI, restoring that exact look.
5. **Snapshot** (floppy icon) — saves a plain image snapshot of the current frame, without the embedded-parameters step.

**Output Upscale interaction:** whatever Output Upscale is set to (Off / 2× / 4× — see below) is applied to both saved snapshots and recorded video, not just the live preview. If you're recording or saving at 4×, the output files come out at that resolution.

---

## Recording Output Video

You can record the generated video output directly from the browser — no screen capture software or OBS needed. The recording reads directly from the diffusion output stream, so it has **no impact on FPS**.

### How to record

1. Start the stream (click **Start Stream**).
2. In the bottom-right corner of the output video, click the **record button** (red circle inside a ring).
3. The button turns into a red **stop square** and a pulsing **REC** badge appears in the top-left corner.
4. When you are done, click the **stop square**.
5. A **Save As** dialog opens — choose a location and filename. The file is saved as `.webm` (plays in any modern browser, VLC, or can be converted with ffmpeg).

### Resolution tip

The recording captures the exact pixel dimensions that the pipeline is outputting. If you want higher-resolution recordings:

- Go to the **Output Upscale** control and set it to **2×** (or 4×) **before** clicking record.
- At 2×, a 512×512 pipeline outputs 1024×1024 to the stream, and that is what gets recorded.
- Changing the upscale setting mid-recording is not recommended — set it first, then record.

### File format

Files are saved as **WebM / VP9**. To convert to MP4:
```
ffmpeg -i recording_20260403_143000.webm -c:v libx264 output.mp4
```

---

## Right Panel — Parameters

These control the internal diffusion process. Hover over any label for a tooltip.

Panel order (top to bottom): **IPAdapter → ControlNet → Timesteps → Streaming Parameters → Temporal Smoothing → LoRAs → Output Upscale.**

---

### IPAdapter Scale

Same slider as in the [IPAdapter section](#loading-a-style-image--ipadapter) above — this is its home on the right panel. See above for the full explanation and value guide.

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

### Temporal Smoothing

A second, separate mechanism for reducing flicker — distinct from ControlNet's Feedback Strength above, though both address the same problem from different angles.

- **Feature Bank** (toggle) — turns temporal smoothing on/off.
- **Blend Weight** (default 0.15) — slider runs from "off" to "smooth." Higher values blend more of the recent feature history into each frame, further reducing flicker at the cost of responsiveness to fast motion.

If you're still seeing flicker after tuning Feedback Strength, this is the next place to look — but try one at a time so you can tell which one is doing the work.

---

### Output Upscale

A fast, simple upscale applied to the output — **Off / 2× / 4×** — separate from the Resolution setting used for generation. This runs after the diffusion pipeline, so it doesn't slow down generation itself, only the final output size.

This affects **saved snapshots and recorded video, not just the live preview** — if Output Upscale is set to 4×, your recordings and snapshots come out at that resolution too. See [Output Controls](#output-controls) above.

---

## How the Main Controls Interact

IPAdapter scale, ControlNet strength, Guidance scale, and any active LoRAs all compete with each other. No single one is fully independent.

| Goal | What to do |
|---|---|
| "More me in the output" | Raise ControlNet strength, lower IPAdapter scale |
| "More style image in the output" | Raise IPAdapter scale, lower ControlNet strength |
| "More prompt-driven output" | Raise Guidance scale, lower IPAdapter scale |
| "Looser, more dreamlike" | Lower Step 1 (toward 20–25), lower ControlNet strength |
| "More stable, less flickery" | Raise Feedback strength slightly (0.25–0.35) |
| "Stronger trained style" | Raise the relevant LoRA's scale; consider lowering IPAdapter scale so they don't fight |

The timestep indices act as a global creativity/fidelity dial underneath all of this — they determine how far into the denoising process each frame is processed, which affects how much any of the other inputs can actually shape the result.

---

## Left Panel — Input Controls

Below the prompt section is the **Input Controls** panel. This maps a physical input to any parameter slider, so you can drive guidance, IPAdapter scale, ControlNet strength, or anything else with your body, breath, or a controller — in real time, without touching the UI. Every input signal can be remapped to a custom min/max range for whatever parameter it's driving, so the same raw signal can be tuned to suit different sliders.

### Hand Gesture
Reads hand gesture from the video input and maps it to a parameter — a way to drive the stream with movement in front of the camera, no separate hardware needed.

### Microphone
Maps the **volume level** of ambient sound or your voice to a parameter. Loud = high value, quiet = low value. Reactive and instant — good for music or voice directly driving a visual effect.

### Gamepad Controller
Maps a gamepad axis or trigger to any parameter. Connect a USB controller and assign axes to whichever sliders you want to control live. **Multimodal** — different parts of the controller (multiple axes, triggers, buttons) can each be mapped to a different parameter at the same time, so one controller can drive several sliders independently.

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

### Noise Floor Calibration (Mic / Breath)

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
| **Prompt 1 / 2 (+)** | Visual style description | — | — |
| **Prompt blend / weights** | Interpolate between prompts (2, or more via + Add Prompt) | — | 0.0–1.0 |
| **IPAdapter Scale** | Style image influence strength | 0.70 | 0.0–1.0 |
| **IPAdapter A→B Blend** | Interpolate between two style images | 0.00 | 0.0–1.0 |
| **LoRA Scale** (per LoRA) | Trained-style influence strength | 0.70 | 0.0–1.0 |
| **ControlNet Strength** | How tightly output tracks camera structure | 0.20 | 0.0–1.0 |
| **Feedback Strength** | How much previous output frame feeds back in | 0.20 | 0.0–1.0 |
| **Temporal Smoothing Blend Weight** | Extra flicker reduction via feature bank | 0.15 | off–smooth |
| **Step 1** | Start of diffusion pass (lower = more creative) | 32 | 0–49 |
| **Step 2** | End of diffusion pass | 45 | 0–49 |
| **Guidance Scale** | Prompt adherence (keep low for streaming) | 1.10 | 1.0–3.0 |
| **Delta** | Noise variation between frames | 0.70 | — |
| **Inference Steps** | Schedule size (leave at 50) | 50 | — |
| **Output Upscale** | Post-process upscale of output/recordings/snapshots | Off | Off / 2× / 4× |
| **Video Speed** | Playback speed of video file input | 1× | 1×–5× |

---

## Technical Details

iVizDiff is built on **[StreamDiffusion](https://github.com/daydreamlive/StreamDiffusion)**, a real-time diffusion pipeline from a 2023 paper by Akio Kodaira et al. ([arXiv:2312.12491](https://arxiv.org/abs/2312.12491)). Standard Stable Diffusion generates one image at a time by running a UNet through a full denoising schedule — too slow for live video. The UNet is still the model doing the actual image transformation at every step; StreamDiffusion doesn't replace it, it restructures how it's called, around four techniques:

- **Stream Batch** — batches the denoising process instead of running it sequentially frame-by-frame, roughly 1.5× faster on its own than sequential denoising.
- **Residual CFG (RCFG)** — the Guidance Scale control in this guide relies on classifier-free guidance, which normally requires an extra UNet pass. RCFG cuts that down to one (or zero) additional passes, up to 2.05× faster than conventional CFG.
- **Stochastic Similarity Filter (SSF)** — skips regenerating a frame when the input hasn't meaningfully changed from the previous one, cutting GPU power draw — measured at 2.39× on an RTX 3060 and 1.99× on an RTX 4090 in the original paper.
- **I/O Queues** — parallelizes input capture, denoising, and output rendering so one stage doesn't stall the others.

Combined with acceleration tooling (TensorRT, Tiny VAE), the paper reports up to 91.07 fps for image-to-image generation on a single RTX 4090 — over 59× the throughput of the baseline Diffusers pipeline it was benchmarked against.

**Note on the repo:** the original StreamDiffusion project (`cumulo-autumn/StreamDiffusion`) has several forks; the one referenced early in this project's history, `livepeer/StreamDiffusion`, is now archived. Active development has moved to [`daydreamlive/StreamDiffusion`](https://github.com/daydreamlive/StreamDiffusion), which is the version linked above.

iVizDiff wraps this pipeline with the live camera/video input, prompt and IPAdapter conditioning, and the parameter controls documented throughout this guide, plus LoRA support, temporal smoothing, output upscaling, and the hand gesture/mic/gamepad/breath input-mapping system layered on top.
