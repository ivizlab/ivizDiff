from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect, UploadFile, File, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from video_input_manager import VideoInputManager
from util import resize_cv_frame
import cv2
import sys
import os

import markdown2

import logging
import uuid
import time
from types import SimpleNamespace
import asyncio
import os
import time
import mimetypes
import torch
import json
import tempfile
from datetime import datetime
from pathlib import Path
from PIL import Image, PngImagePlugin
import yaml

import numpy as np
from config import config, Args
from util import pil_to_frame, pt_to_frame, bytes_to_pil, bytes_to_pt
from connection_manager import ConnectionManager, ServerFullException
from img2img import Pipeline
from input_control import InputManager, GamepadInput, BreathInputSource

# fix mime error on windows
mimetypes.add_type("application/javascript", ".js")

THROTTLE = 1.0 / 120

def load_controlnet_registry():
    """Load ControlNet registry from YAML config file"""
    try:
        registry_path = Path(__file__).parent / "controlnet_registry.yaml"
        with open(registry_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Extract the available_controlnets section
        return config_data.get('available_controlnets', {})
    except Exception as e:
        logging.error(f"load_controlnet_registry: Failed to load ControlNet registry: {e}")
        # Fallback to empty registry
        return {}

def load_default_settings():
    """Load default settings from YAML config file"""
    try:
        registry_path = Path(__file__).parent / "controlnet_registry.yaml"
        with open(registry_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return config_data.get('defaults', {})
    except Exception as e:
        logging.error(f"load_default_settings: Failed to load default settings: {e}")
        # Fallback to hardcoded defaults
        return {
            'guidance_scale': 1.1,
            'delta': 0.7,
            'num_inference_steps': 50,
            'seed': 2,
            't_index_list': [35, 45],
            'ipadapter_scale': 1.0,
            'normalize_prompt_weights': True,
            'normalize_seed_weights': True,
            'prompt': "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
        }

# Load ControlNet registry from config file
AVAILABLE_CONTROLNETS = load_controlnet_registry()
DEFAULT_SETTINGS = load_default_settings()

# Configure logging
def setup_logging(log_level: str = "WARNING"):
    """Setup logging configuration for the application"""
    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.WARNING)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING) 

    # Set up logger for streamdiffusion modules
    streamdiffusion_logger = logging.getLogger('streamdiffusion')
    streamdiffusion_logger.setLevel(numeric_level)
    
    # Set up logger for this application
    app_logger = logging.getLogger('realtime_img2img')
    app_logger.setLevel(numeric_level)
    
    return app_logger

# Initialize logger
logger = setup_logging(config.log_level)


class App:
    def __init__(self, config: Args):
        self.args = config
        self.pipeline = None  # Pipeline created lazily when needed
        self.app = FastAPI()
        self.conn_manager = ConnectionManager()
        self.fps_counter = []
        self.last_fps_update = time.time()
        # Per-phase timing (rolling window, same size as fps_counter)
        self.timing_wait_ms: list = []   # time waiting for frame from frontend
        self.timing_predict_ms: list = []  # time inside pipeline.predict()
        self.timing_encode_ms: list = []   # time encoding tensor → JPEG bytes
        self._last_slow_warn = 0.0  # throttle slow-FPS warnings to once per 5s
        # Store uploaded ControlNet config separately
        self.uploaded_controlnet_config = None
        self.runtime_controlnet_config = None  # Active runtime config (starts from YAML)
        self.config_needs_reload = False  # Track when pipeline needs recreation
        self._pipeline_building = False   # Guards against concurrent pipeline creation
        self.lora_trigger_words: str = ""  # Concatenated trigger words for enabled LoRAs
        # Store current resolution for pipeline recreation
        self.new_width = 512
        self.new_height = 512
        # Store uploaded style images persistently
        self.uploaded_style_image = None
        self.uploaded_style_image_b = None
        # VirtCam subprocess
        self.virtcam_process = None
        self.virtcam_video_path = None
        self.virtcam_speed = 1  # slowdown factor: 1=normal, 2=half speed, …, 5=5× slower
        _repo_root = Path(__file__).parent
        self._virtcam_script = _repo_root / "utils" / "virtcam.py"
        self._source_video_dir = _repo_root / "videos" / "input"
        # Store latest output frame for snapshots
        self.last_output_image: Image.Image | None = None
        # Output upscaling (1 = off, 2 = 2x Lanczos)
        self.upscale_factor: int = 1
        # Constant Frame Rate (CFR) pacing
        self.cfr_enabled: bool = False
        self.target_fps: int = 24
        # Color histogram match post-processing
        self.color_match_enabled: bool = False
        self.color_match_strength: float = 0.4
        self.color_match_reference: np.ndarray | None = None  # BGR uint8
        # Rolling reference: if True, reference slowly tracks output via EMA
        # instead of being locked to a single captured frame
        self.color_match_rolling: bool = True
        self.color_match_adapt_rate: float = 0.02  # ~50 frames to fully adapt
        # Latent feature bank for structural temporal consistency
        self.feature_bank_enabled: bool = False
        self.feature_bank_size: int = 3
        self.feature_bank_weight: float = 0.2
        # Initialize input manager for controller support
        self.input_manager = InputManager()
        # Breath detection source — connects to breath.py WebSocket server
        self.breath_source = BreathInputSource()
        # Initialize video input manager
        self.video_input_manager = VideoInputManager()
        self.video_input_active = False
        self.video_input_mode = "webcam"  # "webcam" or "video_file"
        # LoRA folder — can be overridden by lora_dir in YAML config
        self.lora_dir = Path("C:/AI/models/Lora15")
        self.init_app()
        if self.args.config:
            self._load_yaml_config(self.args.config)

    def _collect_snapshot_params(self) -> dict:
        """Collect the full current parameter state for embedding in a snapshot PNG."""
        state = {}
        if self.pipeline and hasattr(self.pipeline, 'stream') and hasattr(self.pipeline.stream, 'get_stream_state'):
            try:
                state = self.pipeline.stream.get_stream_state()
            except Exception:
                pass

        cfg = self.uploaded_controlnet_config or {}

        # t_index_list
        if self.pipeline and hasattr(self.pipeline, 'stream') and hasattr(self.pipeline.stream, 't_list'):
            t_index_list = [int(v) for v in self.pipeline.stream.t_list]
        elif 't_index_list' in cfg:
            t_index_list = list(cfg['t_index_list'])
        else:
            t_index_list = DEFAULT_SETTINGS.get('t_index_list', [35, 45])

        # scalar stream params
        guidance_scale = state.get('guidance_scale', cfg.get('guidance_scale', DEFAULT_SETTINGS.get('guidance_scale', 1.1)))
        delta = state.get('delta', cfg.get('delta', DEFAULT_SETTINGS.get('delta', 0.7)))
        num_inference_steps = int(state.get('num_inference_steps', cfg.get('num_inference_steps', DEFAULT_SETTINGS.get('num_inference_steps', 50))))

        # seed (prefer first entry from seed_list)
        seed_list_raw = state.get('seed_list') or []
        seed = int(seed_list_raw[0][0]) if seed_list_raw else int(cfg.get('seed', DEFAULT_SETTINGS.get('seed', 2)))

        # ipadapter scale — try live pipeline attribute first, then config
        ipadapter_scale = None
        if self.pipeline and hasattr(self.pipeline, 'stream'):
            raw = getattr(self.pipeline.stream, 'ipadapter_scale', None)
            if raw is not None:
                try:
                    if hasattr(raw, 'item'):
                        ipadapter_scale = float(raw.flatten()[0].item())
                    else:
                        ipadapter_scale = float(raw)
                except Exception:
                    pass
        if ipadapter_scale is None:
            ipadapters = cfg.get('ipadapters', [])
            ipadapter_scale = ipadapters[0].get('scale', DEFAULT_SETTINGS.get('ipadapter_scale', 1.0)) if ipadapters else DEFAULT_SETTINGS.get('ipadapter_scale', 1.0)

        # controlnets — read LIVE values from the pipeline first (reflects slider edits),
        # then overlay model_id / preprocessor info from config dicts
        live_cn = self._get_current_controlnet_config() if self.pipeline else []
        active_cfg = self.runtime_controlnet_config or cfg
        cfg_controlnets = active_cfg.get('controlnets') or []
        controlnets_out = []
        count = max(len(live_cn), len(cfg_controlnets))
        for i in range(count):
            live = live_cn[i] if i < len(live_cn) else {}
            cfgcn = cfg_controlnets[i] if i < len(cfg_controlnets) else {}
            pp = cfgcn.get('preprocessor_params') or {}
            controlnets_out.append({
                'model_id': live.get('model_id') or cfgcn.get('model_id', ''),
                'conditioning_scale': live.get('conditioning_scale', cfgcn.get('conditioning_scale', 1.0)),
                'preprocessor': live.get('preprocessor') or cfgcn.get('preprocessor'),
                'feedback_strength': pp.get('feedback_strength'),
            })

        # prompt_list
        prompt_list = state.get('prompt_list') or []
        if not prompt_list:
            raw_blending = cfg.get('prompt_blending', {})
            if isinstance(raw_blending, dict):
                prompt_list = raw_blending.get('prompt_list', [])
            elif isinstance(raw_blending, list):
                prompt_list = raw_blending

        # seed_list
        seed_list_full = state.get('seed_list') or []
        if not seed_list_full:
            raw_seed = cfg.get('seed_blending', {})
            if isinstance(raw_seed, dict):
                seed_list_full = raw_seed.get('seed_list', [])
            elif isinstance(raw_seed, list):
                seed_list_full = raw_seed

        # ipadapter blend weight — read from inner stream _param_updater
        ipadapter_blend_weight = 0.0
        try:
            inner = self.pipeline.stream.stream if hasattr(self.pipeline.stream, 'stream') else self.pipeline.stream
            ipadapter_blend_weight = float(getattr(inner._param_updater, '_ipadapter_blend_weight', 0.0))
        except Exception:
            pass

        return {
            't_index_list': t_index_list,
            'guidance_scale': guidance_scale,
            'delta': delta,
            'num_inference_steps': num_inference_steps,
            'seed': seed,
            'ipadapter_scale': ipadapter_scale,
            'ipadapter_blend_weight': ipadapter_blend_weight,
            'controlnets': controlnets_out,
            'prompt_list': [[p, w] for p, w in prompt_list],
            'seed_list': [[int(s), w] for s, w in seed_list_full],
        }

    def _load_yaml_config(self, path: str):
        """Load a YAML config file and apply it as if the user clicked Load YAML Config."""
        try:
            config_path = Path(path)
            if not config_path.exists():
                logger.warning(f"_load_yaml_config: Config file not found: {path}")
                return
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            self.uploaded_controlnet_config = config_data
            self.runtime_controlnet_config = None
            self.config_needs_reload = True
            config_width = config_data.get('width', None)
            config_height = config_data.get('height', None)
            if config_width is not None and config_height is not None:
                if config_width % 64 == 0 and config_height % 64 == 0:
                    if 384 <= config_width <= 1024 and 384 <= config_height <= 1024:
                        self.new_width = config_width
                        self.new_height = config_height
            if 'video_input_dir' in config_data:
                self._source_video_dir = Path(config_data['video_input_dir'])
            if 'lora_dir' in config_data:
                self.lora_dir = Path(config_data['lora_dir'])
            if 'cfr_enabled' in config_data:
                self.cfr_enabled = bool(config_data['cfr_enabled'])
            if 'target_fps' in config_data:
                self.target_fps = max(1, int(config_data['target_fps']))
            if 'color_match_enabled' in config_data:
                self.color_match_enabled = bool(config_data['color_match_enabled'])
            if 'color_match_strength' in config_data:
                self.color_match_strength = float(config_data['color_match_strength'])
            if 'color_match_rolling' in config_data:
                self.color_match_rolling = bool(config_data['color_match_rolling'])
            if 'color_match_adapt_rate' in config_data:
                self.color_match_adapt_rate = float(config_data['color_match_adapt_rate'])
            if 'feature_bank_enabled' in config_data:
                self.feature_bank_enabled = bool(config_data['feature_bank_enabled'])
            if 'feature_bank_size' in config_data:
                self.feature_bank_size = max(1, int(config_data['feature_bank_size']))
            if 'feature_bank_weight' in config_data:
                self.feature_bank_weight = float(config_data['feature_bank_weight'])
            logger.info(f"_load_yaml_config: Loaded config from {path}")
        except Exception as e:
            logger.error(f"_load_yaml_config: Failed to load config {path}: {e}")

    def cleanup(self):
        """Cleanup resources when app is shutting down"""
        logger.info("App cleanup: Starting application cleanup...")
        # Stop video input
        if self.video_input_manager:
            self.video_input_manager.stop()
        if self.pipeline:
            self._cleanup_pipeline(self.pipeline)
            self.pipeline = None
        logger.info("App cleanup: Completed application cleanup")
        if self.pipeline:
            self._cleanup_pipeline(self.pipeline)
            self.pipeline = None
        logger.info("App cleanup: Completed application cleanup")

    async def _async_pipeline_update(self, **kwargs) -> None:
        """Run pipeline.update_stream_params in a thread so HTTP endpoints don't block the event loop."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self.pipeline.update_stream_params(**kwargs))

    def _handle_input_parameter_update(self, parameter_name: str, value: float) -> None:
        """Handle parameter updates from input controls"""
        try:
            if not self.pipeline or not hasattr(self.pipeline, 'stream'):
                logger.warning(f"_handle_input_parameter_update: No pipeline available for parameter {parameter_name}")
                return

            # Map parameter names to pipeline update methods
            if parameter_name == 'guidance_scale':
                self.pipeline.update_stream_params(guidance_scale=value)
            elif parameter_name == 'delta':
                self.pipeline.update_stream_params(delta=value)
            elif parameter_name == 'num_inference_steps':
                self.pipeline.update_stream_params(num_inference_steps=int(value))
            elif parameter_name == 'seed':
                self.pipeline.update_stream_params(seed=int(value))
            elif parameter_name == 'ipadapter_scale':
                self.pipeline.update_stream_params(ipadapter_config={'scale': value})
            elif parameter_name == 'ipadapter_weight_type':
                # For weight type, we need to convert the numeric value to a string
                weight_types = ["linear", "ease in", "ease out", "ease in-out", "reverse in-out", 
                               "weak input", "weak output", "weak middle", "strong middle", 
                               "style transfer", "composition", "strong style transfer", 
                               "style and composition", "style transfer precise", "composition precise"]
                index = int(value) % len(weight_types)
                self.pipeline.update_ipadapter_weight_type(weight_types[index])
            elif parameter_name.startswith('controlnet_') and parameter_name.endswith('_strength'):
                # Handle ControlNet strength parameters
                import re
                match = re.match(r'controlnet_(\d+)_strength', parameter_name)
                if match:
                    index = int(match.group(1))
                    # Use existing ControlNet strength update logic
                    current_config = self._get_current_controlnet_config()
                    if current_config and index < len(current_config):
                        current_config[index]['conditioning_scale'] = float(value)
                        # Apply the updated config via unified API
                        self.pipeline.update_stream_params(controlnet_config=current_config)
            elif parameter_name.startswith('controlnet_') and '_preprocessor_' in parameter_name:
                # Handle ControlNet preprocessor parameters
                match = re.match(r'controlnet_(\d+)_preprocessor_(.+)', parameter_name)
                if match:
                    controlnet_index = int(match.group(1))
                    param_name = match.group(2)
                    # Use the same approach as the API endpoint
                    current_config = self._get_current_controlnet_config()
                    if current_config and controlnet_index < len(current_config):
                        # Update preprocessor_params for the specified controlnet
                        if 'preprocessor_params' not in current_config[controlnet_index]:
                            current_config[controlnet_index]['preprocessor_params'] = {}
                        current_config[controlnet_index]['preprocessor_params'][param_name] = value
                        self.pipeline.update_stream_params(controlnet_config=current_config)
            elif parameter_name.startswith('prompt_weight_'):
                # Handle prompt blending weights
                match = re.match(r'prompt_weight_(\d+)', parameter_name)
                if match:
                    index = int(match.group(1))
                    # Get current prompt list from unified state and update specific weight
                    state = self.pipeline.stream.get_stream_state()
                    current_prompts = state.get('prompt_list', [])
                    if current_prompts and index < len(current_prompts):
                        updated_prompts = list(current_prompts)
                        updated_prompts[index] = (updated_prompts[index][0], float(value))
                        self.pipeline.update_stream_params(prompt_list=updated_prompts)
            elif parameter_name.startswith('seed_weight_'):
                # Handle seed blending weights  
                match = re.match(r'seed_weight_(\d+)', parameter_name)
                if match:
                    index = int(match.group(1))
                    # Get current seed list from unified state and update specific weight
                    state = self.pipeline.stream.get_stream_state()
                    current_seeds = state.get('seed_list', [])
                    if current_seeds and index < len(current_seeds):
                        updated_seeds = list(current_seeds)
                        updated_seeds[index] = (updated_seeds[index][0], float(value))
                        self.pipeline.update_stream_params(seed_list=updated_seeds)
            else:
                logger.warning(f"_handle_input_parameter_update: Unknown parameter {parameter_name}")

            logger.info(f"_handle_input_parameter_update: Updated {parameter_name} to {value}")
        except Exception as e:
            logger.error(f"_handle_input_parameter_update: Failed to update {parameter_name}: {e}")


    


    def _get_controlnet_pipeline(self):
        """Get the ControlNet pipeline from the main pipeline structure"""
        if not self.pipeline:
            return None
            
        stream = self.pipeline.stream
        
        # Module-aware: module installs expose preprocessors on stream
        if hasattr(stream, 'preprocessors'):
            return stream
            
        # Check if stream has nested stream (IPAdapter wrapper)
        if hasattr(stream, 'stream') and hasattr(stream.stream, 'preprocessors'):
            return stream.stream
            
        # New module path on stream
        if hasattr(stream, '_controlnet_module'):
            return stream._controlnet_module
        return None

    def _get_current_controlnet_config(self):
        """Get the current ControlNet configuration state from the pipeline"""
        cn_pipeline = self._get_controlnet_pipeline()
        if not cn_pipeline or not hasattr(cn_pipeline, 'controlnets'):
            return []
        
        current_config = []
        for i, controlnet in enumerate(cn_pipeline.controlnets):
            model_id = getattr(controlnet, 'model_id', f'controlnet_{i}')
            scale = cn_pipeline.controlnet_scales[i] if hasattr(cn_pipeline, 'controlnet_scales') and i < len(cn_pipeline.controlnet_scales) else 1.0
            
            config = {
                'model_id': model_id,
                'conditioning_scale': scale,
                'preprocessor': getattr(cn_pipeline.preprocessors[i], '__class__.__name__', '').replace('Preprocessor', '').lower() if cn_pipeline.preprocessors[i] else None,
                'enabled': True,
                'preprocessor_params': getattr(cn_pipeline.preprocessors[i], 'params', {}) if cn_pipeline.preprocessors[i] else {}
            }

            current_config.append(config)
        return current_config

    def _get_inner_stream(self):
        """Return the inner StreamDiffusion instance, or None if not available."""
        try:
            s = self.pipeline.stream
            return s.stream if hasattr(s, 'stream') else s
        except Exception:
            return None

    def _postprocess_frame(self, image, output_type: str) -> bytes:
        """
        Run all CPU-heavy post-processing in a thread executor:
          1. Color histogram match (cv2, EMA reference update)
          2. Snapshot capture (for /api/snapshot)
          3. Upscale (optional Lanczos)
          4. JPEG encode → multipart bytes

        Called via loop.run_in_executor so the event loop is never blocked.
        """
        # --- Color histogram match ---
        if self.color_match_enabled:
            try:
                if output_type == "pt":
                    _t = image[0] if image.dim() == 4 else image
                    _src_rgb = (_t.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
                else:
                    _src_rgb = np.array(image)
                _src_bgr = cv2.cvtColor(_src_rgb, cv2.COLOR_RGB2BGR)

                if self.color_match_rolling:
                    if self.color_match_reference is None:
                        self.color_match_reference = _src_bgr.astype("float32")
                    else:
                        r = self.color_match_adapt_rate
                        self.color_match_reference = (
                            (1.0 - r) * self.color_match_reference
                            + r * _src_bgr.astype("float32")
                        )

                if self.color_match_reference is not None:
                    _ref = np.clip(self.color_match_reference, 0, 255).astype("uint8")
                    _matched_bgr = self._apply_color_match(_src_bgr, _ref, self.color_match_strength)
                    _matched_rgb = cv2.cvtColor(_matched_bgr, cv2.COLOR_BGR2RGB)
                    if output_type == "pt":
                        _matched_t = torch.from_numpy(_matched_rgb.astype("float32") / 255.0).permute(2, 0, 1)
                        image = _matched_t.unsqueeze(0).to(_t.device, _t.dtype) if image.dim() == 4 else _matched_t.to(_t.device, _t.dtype)
                    else:
                        image = Image.fromarray(_matched_rgb)
            except Exception as _cm_err:
                logger.warning(f"color_match: {_cm_err}")

        # --- Snapshot cache ---
        try:
            if output_type == "pt":
                t = image[0] if image.dim() == 4 else image
                t_uint8 = (t * 255).clamp(0, 255).to(torch.uint8)
                self.last_output_image = Image.fromarray(t_uint8.permute(1, 2, 0).cpu().numpy())
            else:
                self.last_output_image = image.copy() if hasattr(image, 'copy') else image
        except Exception as _snap_err:
            logger.warning(f"snapshot capture failed: {_snap_err}")

        # --- Upscale + JPEG encode ---
        if self.upscale_factor > 1:
            if output_type == "pt":
                t = image[0] if image.dim() == 4 else image
                pil_image = Image.fromarray(
                    (t.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
                )
            else:
                pil_image = image
            w, h = pil_image.size
            pil_image = pil_image.resize(
                (w * self.upscale_factor, h * self.upscale_factor), Image.LANCZOS
            )
            return pil_to_frame(pil_image)
        elif output_type == "pt":
            return pt_to_frame(image)
        else:
            return pil_to_frame(image)

    def _apply_feature_bank_settings(self) -> None:
        """Push current feature bank settings into the live pipeline."""
        sd = self._get_inner_stream()
        if sd is None:
            return
        sd.feature_bank_enabled = self.feature_bank_enabled
        sd.feature_bank_weight = self.feature_bank_weight
        if hasattr(sd, 'set_feature_bank_size'):
            sd.set_feature_bank_size(self.feature_bank_size)

    def _post_pipeline_build(self) -> None:
        """Run once after every pipeline build or rebuild.

        1. Push feature-bank settings so YAML feature_bank_enabled actually takes effect.
        2. Re-run _recalculate_timestep_dependent_params with the pipeline's own t_list —
           this seeds _desired_t_index_list in the fresh StreamParameterUpdater, which
           prevents the 'Step 1 nudge needed after LoRA apply' regression where the
           denoising tensors are in a partially-uninitialised state until the user
           manually moves a timestep slider.
        3. Reset colour-match reference so the new session isn't anchored to the look
           of the old session.
        """
        self._apply_feature_bank_settings()
        self.color_match_reference = None
        try:
            sd = self._get_inner_stream()
            if sd is not None and hasattr(sd, 't_list') and sd.t_list:
                self.pipeline.update_stream_params(t_index_list=list(sd.t_list))
        except Exception as _e:
            logger.warning(f"_post_pipeline_build: t_index_list re-apply failed: {_e}")

    def _apply_color_match(self, source_bgr: np.ndarray, reference_bgr: np.ndarray, strength: float) -> np.ndarray:
        """
        Match the color cast of source to reference using mean/std transfer in LAB space.
        Only the A and B channels (color) are matched — L (luminance) is left unchanged
        so brightness dynamics are preserved.
        strength: 0.0 = no change, 1.0 = full match.
        """
        src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype("float32")
        ref_lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB).astype("float32")

        matched = src_lab.copy()
        # Channels 1 (A) and 2 (B) carry warm/cool and green/magenta cast
        for ch in (1, 2):
            s_mean = src_lab[:, :, ch].mean()
            s_std  = src_lab[:, :, ch].std()
            r_mean = ref_lab[:, :, ch].mean()
            r_std  = ref_lab[:, :, ch].std()
            if s_std > 1e-6:
                matched[:, :, ch] = (src_lab[:, :, ch] - s_mean) * (r_std / (s_std + 1e-6)) + r_mean

        matched = np.clip(matched, 0, 255).astype("uint8")
        result_bgr = cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)

        if strength >= 1.0:
            return result_bgr
        return cv2.addWeighted(source_bgr, 1.0 - strength, result_bgr, strength, 0)

    def init_app(self):
        # Enhanced CORS for API-only development mode
        if self.args.api_only:
            # More permissive CORS for development
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],  # Include common Vite dev ports
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        else:
            # Standard CORS for production
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Set up input manager callback for parameter updates
        self.input_manager.set_parameter_update_callback(self._handle_input_parameter_update)

        @self.app.on_event("startup")
        async def _start_breath_source():
            await self.breath_source.start()

        @self.app.on_event("shutdown")
        async def _stop_breath_source():
            await self.breath_source.stop()

        @self.app.get("/api/breath/status")
        async def get_breath_status():
            """Return current breath.py connection state and normalised signal values.

            Response:
              status     — "DISCONNECTED" | "CONNECTED" | "LIVE"
              last_event — most recent event type: "inhale", "exhale", "cycle_end", or ""
              amplitude  — rolling-max normalised breath amplitude, 0.0–1.0
              bpm        — bpm_rolling normalised over 5–30 BPM range, 0.0–1.0
            """
            return JSONResponse(self.breath_source.get_status())

        @self.app.websocket("/api/ws/{user_id}")
        async def websocket_endpoint(user_id: uuid.UUID, websocket: WebSocket):
            try:
                await self.conn_manager.connect(
                    user_id, websocket, self.args.max_queue_size
                )
                await handle_websocket_data(user_id)
            except ServerFullException as e:
                logging.error(f"Server Full: {e}")
            finally:
                await self.conn_manager.disconnect(user_id)
                logging.info(f"User disconnected: {user_id}")

        async def handle_websocket_data(user_id: uuid.UUID):
            if not self.conn_manager.check_user(user_id):
                return HTTPException(status_code=404, detail="User not found")
            last_time = time.time()
            _ws_loop = asyncio.get_event_loop()
            try:
                while True:
                    if (
                        self.args.timeout > 0
                        and time.time() - last_time > self.args.timeout
                    ):
                        await self.conn_manager.send_json(
                            user_id,
                            {
                                "status": "timeout",
                                "message": "Your session has ended",
                            },
                        )
                        await self.conn_manager.disconnect(user_id)
                        return
                    data = await self.conn_manager.receive_json(user_id)
                    # logger.info(f"WebSocket DEBUG: data type={type(data)}, content preview={str(data)[:100] if data else 'None'}")
                    if data is None:
                        break
                   # Inside handle_websocket_data() in main.py
                    if data["status"] == "next_frame":
                        # This sequence is correct: receive status, then params.
                        params_data = await self.conn_manager.receive_json(user_id)
                        if params_data is None:
                            break # Exit if the connection is closed while receiving params
                            
                        params = Pipeline.InputParams(**params_data)
                        params = SimpleNamespace(**params.dict())
                        
                        # Determine if an image is needed for the pipeline
                        need_image = True
                        if self.pipeline and hasattr(self.pipeline, 'pipeline_mode'):
                            has_controlnets = self.pipeline.use_config and self.pipeline.config and 'controlnets' in self.pipeline.config
                            need_image = self.pipeline.pipeline_mode == "img2img" or has_controlnets
                        elif self.uploaded_controlnet_config and 'mode' in self.uploaded_controlnet_config:
                            has_controlnets = 'controlnets' in self.uploaded_controlnet_config
                            need_image = self.uploaded_controlnet_config['mode'] == "img2img" or has_controlnets


                        if need_image:
                            image_data = await self.conn_manager.receive_bytes(user_id)

                            if self.video_input_mode == "video_file" and self.video_input_active:
                                # logger.info(f"VIDEO: Using video input: mode={self.video_input_mode}, active={self.video_input_active}")
                                frame = self.video_input_manager.get_current_frame()
                                if frame is not None:
                                    # logger.info(f"VIDEO: Got video frame: {frame.shape}")

                                    # --- START of FIX ---
                                    # Import the resize function at the top of your main.py if it's not there
                                    # from util import resize_cv_frame

                                    # Resize the frame to match the current pipeline resolution before processing.
                                    # We use maintain_aspect_ratio=True to prevent distortion.
                                    resized_frame = resize_cv_frame(
                                        frame, 
                                        self.new_width, 
                                        self.new_height, 
                                        maintain_aspect_ratio=True
                                    )
                                    
                                    # Use the *resized* frame for the rest of the process.
                                    frame_bytes = self.video_input_manager.frame_to_bytes(resized_frame)
                                    # --- END of FIX ---

                                    params.image = await _ws_loop.run_in_executor(None, bytes_to_pt, frame_bytes)
                                else:
                                    logger.info(f"VIDEO: No video frame available, falling back to client image.")
                                    if not image_data:
                                        await self.conn_manager.send_json(user_id, {"status": "send_frame"})
                                        continue
                                    params.image = await _ws_loop.run_in_executor(None, bytes_to_pt, image_data)
                            else:
                                # Default to using the client's webcam stream
                                if not image_data:
                                    await self.conn_manager.send_json(user_id, {"status": "send_frame"})
                                    continue
                                params.image = await _ws_loop.run_in_executor(None, bytes_to_pt, image_data)

                        else:
                            params.image = None
                        
                        await self.conn_manager.update_data(user_id, params)

            except Exception as e:
                logging.error(f"Websocket Error: {e}, {user_id} ")
                await self.conn_manager.disconnect(user_id)

        @self.app.get("/api/queue")
        async def get_queue_size():
            queue_size = self.conn_manager.get_user_count()
            return JSONResponse({"queue_size": queue_size})

        @self.app.get("/api/stream/{user_id}")
        async def stream(user_id: uuid.UUID, request: Request):
            try:
                _loop = asyncio.get_event_loop()

                async def _ensure_pipeline_built(build_fn):
                    """Run blocking pipeline build in a thread; event loop stays responsive.
                    If another build is already running, waits for it to finish."""
                    if self._pipeline_building:
                        logger.warning("stream: pipeline build already in progress — waiting")
                        while self._pipeline_building:
                            await asyncio.sleep(0.5)
                        return
                    self._pipeline_building = True
                    try:
                        self.pipeline = await _loop.run_in_executor(None, build_fn)
                    finally:
                        self._pipeline_building = False

                async def generate():
                    loop = asyncio.get_event_loop()
                    logger.info(f"generate: started for user {user_id} | pipeline={self.pipeline is not None} config_needs_reload={self.config_needs_reload}")

                    # ── Pipeline setup ──────────────────────────────────────────────────
                    # This runs INSIDE the generator so that StreamingResponse is returned
                    # immediately. HTTP 200 + multipart headers reach the browser before
                    # any model loading begins, preventing browser timeout during rebuild.
                    try:
                        if self.pipeline is None:
                            build_label = "ControlNet" if self.uploaded_controlnet_config else "default"
                            logger.info(f"stream: Creating {build_label} pipeline...")
                            build_fn = self._create_pipeline_with_config if self.uploaded_controlnet_config else self._create_default_pipeline
                            await _ensure_pipeline_built(build_fn)
                            self.config_needs_reload = False
                            self._post_pipeline_build()
                            logger.info("stream: Pipeline created successfully")
                            try:
                                acc = getattr(self.args, 'acceleration', None)
                                logger.debug(f"stream: acceleration={acc}, use_config={getattr(self.pipeline, 'use_config', False)}")
                                stream_obj = getattr(self.pipeline, 'stream', None)
                                unet_obj = getattr(stream_obj, 'unet', None)
                                is_trt = unet_obj is not None and hasattr(unet_obj, 'engine') and hasattr(unet_obj, 'stream')
                                logger.debug(f"stream: unet_is_trt={is_trt}, has_ipadapter={getattr(self.pipeline, 'has_ipadapter', False)}")
                            except Exception:
                                logger.exception("stream: failed to log pipeline state after creation")

                        elif self.config_needs_reload or (self.uploaded_controlnet_config and not (self.pipeline.use_config and self.pipeline.config and 'controlnets' in self.pipeline.config)) or (self.uploaded_controlnet_config and not self.pipeline.use_config):
                            loras = (self.uploaded_controlnet_config or {}).get('loras', [])
                            logger.info(f"stream: Recreating pipeline (LoRA/config change) | loras={len(loras)} ...")
                            old_pipeline = self.pipeline
                            self.pipeline = None
                            if old_pipeline:
                                logger.info("stream: Cleaning up old pipeline...")
                                await loop.run_in_executor(None, self._cleanup_pipeline, old_pipeline)
                                logger.info("stream: Old pipeline cleanup done")
                            build_fn = self._create_pipeline_with_config if self.uploaded_controlnet_config else self._create_default_pipeline
                            logger.info(f"stream: Building new pipeline with {build_fn.__name__} ...")
                            await _ensure_pipeline_built(build_fn)
                            self.config_needs_reload = False
                            self._post_pipeline_build()
                            logger.info(f"stream: Pipeline recreated successfully | pipeline={self.pipeline is not None}")

                    except Exception:
                        logger.exception("stream: pipeline build/recreate failed — aborting stream")
                        return  # closes the generator; browser sees end-of-multipart

                    # ── Frame generation loop ────────────────────────────────────────
                    # Kick off the very first frame request before entering the loop.
                    await self.conn_manager.send_json(user_id, {"status": "send_frame"})

                    # CFR: track when the PREVIOUS frame was yielded.
                    # Each iteration we sleep only the remaining gap to hit target period.
                    # If processing already exceeded the period, sleep = 0 (no catch-up).
                    _cfr_last_yield = time.perf_counter()

                    while True:
                        frame_start_time = time.time()

                        # --- Phase 1: wait for frame the frontend already started sending ---
                        params = await self.conn_manager.get_latest_data(user_id)
                        t_after_wait = time.time()
                        if params is None:
                            # If the user is gone (disconnected), stop this generator.
                            # Without this check the coroutine either hangs forever at
                            # queue.get() or spins in a tight loop starving the event loop.
                            if not self.conn_manager.check_user(user_id):
                                logger.info(f"generate: user {user_id} disconnected — stopping stream")
                                return
                            # No frame yet — re-request and retry
                            await self.conn_manager.send_json(
                                user_id, {"status": "send_frame"}
                            )
                            continue

                        # --- Pipeline: ask for the NEXT frame before starting inference ---
                        # The frontend now prepares the next webcam frame while the GPU works,
                        # overlapping frontend capture with inference and eliminating wait time.
                        await self.conn_manager.send_json(user_id, {"status": "send_frame"})

                        try:
                            # --- Phase 2: inference ---
                            # Run in executor so the event loop stays free for
                            # handle_websocket_data() to receive the next frame concurrently.
                            image = await loop.run_in_executor(
                                None, self.pipeline.predict, params
                            )
                            t_after_predict = time.time()
                            if image is None:
                                logger.error("generate: predict returned None image; skipping frame")
                                continue

                            # --- Phase 3: post-process + encode (in executor, non-blocking) ---
                            # Color match, snapshot cache, upscale, JPEG encode all run
                            # in a thread so the event loop stays free for WebSocket I/O.
                            frame = await loop.run_in_executor(
                                None, self._postprocess_frame, image, self.pipeline.output_type
                            )
                            t_after_encode = time.time()
                        except Exception as e:
                            logger.exception(f"generate: predict failed with exception: {e}")
                            continue

                        # --- Per-phase timing (processing only, excludes CFR sleep) ---
                        _WINDOW = 30
                        frame_time = t_after_encode - frame_start_time
                        wait_ms   = (t_after_wait    - frame_start_time) * 1000
                        pred_ms   = (t_after_predict  - t_after_wait)    * 1000
                        enc_ms    = (t_after_encode   - t_after_predict) * 1000

                        self.timing_wait_ms.append(wait_ms)
                        self.timing_predict_ms.append(pred_ms)
                        self.timing_encode_ms.append(enc_ms)

                        # --- CFR pacing ---
                        # Sleep only the remaining gap between last yield and next target.
                        # If processing already took longer than the period, sleep = 0.
                        _cfr_on = self.cfr_enabled
                        _cfr_sleep = 0.0
                        if _cfr_on:
                            _frame_period = 1.0 / max(1, self.target_fps)
                            _cfr_sleep = (_cfr_last_yield + _frame_period) - time.perf_counter()
                            if _cfr_sleep > 0.0:
                                await asyncio.sleep(_cfr_sleep)

                        # --- fps_counter: delivery cadence (includes CFR sleep) ---
                        t_after_cfr = time.time()
                        delivery_time = t_after_cfr - frame_start_time
                        self.fps_counter.append(delivery_time)
                        if len(self.fps_counter) > _WINDOW:
                            self.fps_counter.pop(0)
                            self.timing_wait_ms.pop(0)
                            self.timing_predict_ms.pop(0)
                            self.timing_encode_ms.pop(0)

                        # --- Auto-warn on slow GPU processing (throttled to once per 5 s) ---
                        fps_now = 1.0 / frame_time if frame_time > 0 else 0
                        _now = time.time()
                        if fps_now < 20 and (_now - self._last_slow_warn) > 5.0:
                            self._last_slow_warn = _now
                            _cuda_info = ""
                            if torch.cuda.is_available():
                                alloc = torch.cuda.memory_allocated() / 1e9
                                reserv = torch.cuda.memory_reserved() / 1e9
                                _cuda_info = f" | CUDA alloc={alloc:.2f}GB reserved={reserv:.2f}GB"
                            logger.warning(
                                f"PERF DROP: {fps_now:.1f} fps (gpu) | "
                                f"wait={wait_ms:.0f}ms predict={pred_ms:.0f}ms encode={enc_ms:.0f}ms"
                                f"{_cuda_info}"
                            )

                        yield frame
                        _cfr_last_yield = time.perf_counter()  # record actual yield time
                        if self.args.debug:
                            logger.debug(f"gpu={frame_time*1000:.1f}ms delivery={delivery_time*1000:.1f}ms cfr_sleep={_cfr_sleep*1000:.0f}ms (wait={wait_ms:.0f} pred={pred_ms:.0f} enc={enc_ms:.0f})")

                return StreamingResponse(
                    generate(),
                    media_type="multipart/x-mixed-replace;boundary=frame",
                    headers={"Cache-Control": "no-cache"},
                )
            except Exception as e:
                logging.error(f"Streaming Error: {e}, {user_id} ")
                return HTTPException(status_code=404, detail="User not found")

        # route to setup frontend
        @self.app.get("/api/settings")
        async def settings():
            # Use Pipeline class directly for schema info (doesn't require instance)
            info_schema = Pipeline.Info.schema()
            info = Pipeline.Info()
            if info.page_content:
                page_content = markdown2.markdown(info.page_content)

            input_params = Pipeline.InputParams.schema()
            
            # Add ControlNet information 
            controlnet_info = self._get_controlnet_info()
            
            # Add IPAdapter information
            ipadapter_info = self._get_ipadapter_info()
            
            # Include config prompt if available, otherwise use default
            config_prompt = None
            if self.uploaded_controlnet_config and 'prompt' in self.uploaded_controlnet_config:
                config_prompt = self.uploaded_controlnet_config['prompt']
            elif not config_prompt:
                config_prompt = DEFAULT_SETTINGS.get('prompt')
            
            # Get current t_index_list from pipeline or config
            current_t_index_list = None
            if self.pipeline and hasattr(self.pipeline.stream, 't_list'):
                current_t_index_list = self.pipeline.stream.t_list
            elif self.uploaded_controlnet_config and 't_index_list' in self.uploaded_controlnet_config:
                current_t_index_list = self.uploaded_controlnet_config['t_index_list']
            else:
                # Default values
                current_t_index_list = DEFAULT_SETTINGS.get('t_index_list', [35, 45])
            
            # Get current acceleration setting
            current_acceleration = self.args.acceleration
            
            # Get current resolution
            current_resolution = f"{self.new_width}x{self.new_height}"
            # Add aspect ratio for display
            aspect_ratio = self._calculate_aspect_ratio(self.new_width, self.new_height)
            if aspect_ratio:
                current_resolution += f" ({aspect_ratio})"
            if self.uploaded_controlnet_config and 'acceleration' in self.uploaded_controlnet_config:
                current_acceleration = self.uploaded_controlnet_config['acceleration']
            
            # Get current streaming parameters (default values or from pipeline if available)
            current_guidance_scale = DEFAULT_SETTINGS.get('guidance_scale', 1.1)
            current_delta = DEFAULT_SETTINGS.get('delta', 0.7)
            current_num_inference_steps = DEFAULT_SETTINGS.get('num_inference_steps', 50)
            current_seed = DEFAULT_SETTINGS.get('seed', 2)
            
            if self.pipeline and hasattr(self.pipeline.stream, 'get_stream_state'):
                state = self.pipeline.stream.get_stream_state()
                current_guidance_scale = state.get('guidance_scale', DEFAULT_SETTINGS.get('guidance_scale', 1.1))
                current_delta = state.get('delta', DEFAULT_SETTINGS.get('delta', 0.7))
                current_num_inference_steps = state.get('num_inference_steps', DEFAULT_SETTINGS.get('num_inference_steps', 50))
                current_seed = state.get('current_seed', DEFAULT_SETTINGS.get('seed', 2))
            elif self.uploaded_controlnet_config:
                current_guidance_scale = self.uploaded_controlnet_config.get('guidance_scale', DEFAULT_SETTINGS.get('guidance_scale', 1.1))
                current_delta = self.uploaded_controlnet_config.get('delta', DEFAULT_SETTINGS.get('delta', 0.7))
                current_num_inference_steps = self.uploaded_controlnet_config.get('num_inference_steps', DEFAULT_SETTINGS.get('num_inference_steps', 50))
                current_seed = self.uploaded_controlnet_config.get('seed', DEFAULT_SETTINGS.get('seed', 2))
            
            # Get prompt and seed blending configuration from uploaded config or pipeline
            prompt_blending_config = None
            seed_blending_config = None
            
            # First try to get from current pipeline if available
            if self.pipeline and hasattr(self.pipeline.stream, 'get_stream_state'):
                state = self.pipeline.stream.get_stream_state()
                current_prompts = state.get('prompt_list', [])
                current_seeds = state.get('seed_list', [])
                if current_prompts:
                    prompt_blending_config = current_prompts
                if current_seeds:
                    seed_blending_config = current_seeds
            
            # If not available from pipeline, get from uploaded config and normalize
            if not prompt_blending_config:
                prompt_blending_config = self._normalize_prompt_config(self.uploaded_controlnet_config)
            
            if not seed_blending_config:
                seed_blending_config = self._normalize_seed_config(self.uploaded_controlnet_config)
            
            # Get current normalize weights settings
            normalize_prompt_weights = True  # default
            normalize_seed_weights = True    # default
            
            if self.pipeline and hasattr(self.pipeline.stream, 'get_stream_state'):
                state = self.pipeline.stream.get_stream_state()
                normalize_prompt_weights = state.get('normalize_prompt_weights', True)
                normalize_seed_weights = state.get('normalize_seed_weights', True)
            elif self.uploaded_controlnet_config:
                normalize_prompt_weights = self.uploaded_controlnet_config.get('normalize_weights', True)
                normalize_seed_weights = self.uploaded_controlnet_config.get('normalize_weights', True)
            
            # Get current negative prompt
            config_negative_prompt = None
            if self.pipeline and hasattr(self.pipeline, 'negative_prompt'):
                config_negative_prompt = self.pipeline.negative_prompt
            elif self.uploaded_controlnet_config and 'negative_prompt' in self.uploaded_controlnet_config:
                config_negative_prompt = self.uploaded_controlnet_config['negative_prompt']
            else:
                config_negative_prompt = DEFAULT_SETTINGS.get('negative_prompt', '')

            return JSONResponse(
                {
                    "info": info_schema,
                    "input_params": input_params,
                    "max_queue_size": self.args.max_queue_size,
                    "page_content": page_content if info.page_content else "",
                    "controlnet": controlnet_info,
                    "ipadapter": ipadapter_info,
                    "config_prompt": config_prompt,
                    "config_negative_prompt": config_negative_prompt,
                    "t_index_list": current_t_index_list,
                    "acceleration": current_acceleration,
                    "guidance_scale": current_guidance_scale,
                    "delta": current_delta,
                    "num_inference_steps": current_num_inference_steps,
                    "seed": current_seed,
                    "current_resolution": current_resolution,
                    "prompt_blending": prompt_blending_config,
                    "seed_blending": seed_blending_config,
                    "normalize_prompt_weights": normalize_prompt_weights,
                    "normalize_seed_weights": normalize_seed_weights,
                    "cfr_enabled": self.cfr_enabled,
                    "target_fps": self.target_fps,
                    "color_match_enabled": self.color_match_enabled,
                    "color_match_strength": self.color_match_strength,
                    "color_match_has_reference": self.color_match_reference is not None,
                    "feature_bank_enabled": self.feature_bank_enabled,
                    "feature_bank_size": self.feature_bank_size,
                    "feature_bank_weight": self.feature_bank_weight,
                    "loras": (self.uploaded_controlnet_config or {}).get("loras", []),
                    "lora_dir": str(self.lora_dir).replace("\\", "/"),
                }
            )

        @self.app.post("/api/controlnet/upload-config")
        async def upload_controlnet_config(file: UploadFile = File(...)):
            """Upload and load a new ControlNet YAML configuration"""
            try:
                if not file.filename.endswith(('.yaml', '.yml')):
                    raise HTTPException(status_code=400, detail="File must be a YAML file")
                
                # Save uploaded file temporarily
                content = await file.read()
                
                # Parse YAML content
                try:
                    config_data = yaml.safe_load(content.decode('utf-8'))
                except yaml.YAMLError as e:
                    raise HTTPException(status_code=400, detail=f"Invalid YAML format: {str(e)}")
                
                # YAML is source of truth - completely replace any runtime modifications
                self.uploaded_controlnet_config = config_data
                self.runtime_controlnet_config = None  # Clear any runtime additions
                self.config_needs_reload = True  # Mark that pipeline needs recreation
                
                logger.info(f"upload_controlnet_config: YAML uploaded - resetting ControlNet configuration to source of truth")
                
                # Log IPAdapter configuration for debugging
    
                
                # Get config prompt if available
                config_prompt = config_data.get('prompt', None)
                
                # Get t_index_list from config if available
                t_index_list = config_data.get('t_index_list', DEFAULT_SETTINGS.get('t_index_list', [35, 45]))
                
                # Get acceleration from config if available
                config_acceleration = config_data.get('acceleration', self.args.acceleration)
                
                # Get width and height from config if available
                config_width = config_data.get('width', None)
                config_height = config_data.get('height', None)
                
                # Update resolution if width/height are specified in config
                if config_width is not None and config_height is not None:
                    try:
                        # Validate resolution
                        if config_width % 64 != 0 or config_height % 64 != 0:
                            raise HTTPException(status_code=400, detail="Resolution must be multiples of 64")
                        
                        if not (384 <= config_width <= 1024) or not (384 <= config_height <= 1024):
                            raise HTTPException(status_code=400, detail="Resolution must be between 384 and 1024")
                        
                        # Update the resolution
                        self.new_width = config_width
                        self.new_height = config_height
                        logger.info(f"upload_controlnet_config: Updated resolution to {config_width}x{config_height}")
                    except Exception as e:
                        logging.error(f"upload_controlnet_config: Failed to update resolution: {e}")
                        # Don't fail the upload, just log the error
                
                # Update video input directory if specified in config
                if 'video_input_dir' in config_data:
                    self._source_video_dir = Path(config_data['video_input_dir'])
                    logger.info(f"upload_controlnet_config: video_input_dir set to {self._source_video_dir}")

                # Update CFR settings if specified in config
                if 'cfr_enabled' in config_data:
                    self.cfr_enabled = bool(config_data['cfr_enabled'])
                if 'target_fps' in config_data:
                    self.target_fps = max(1, int(config_data['target_fps']))
                # Update color match settings if specified in config
                if 'color_match_enabled' in config_data:
                    self.color_match_enabled = bool(config_data['color_match_enabled'])
                if 'color_match_strength' in config_data:
                    self.color_match_strength = float(config_data['color_match_strength'])
                # Update feature bank settings if specified in config
                if 'feature_bank_enabled' in config_data:
                    self.feature_bank_enabled = bool(config_data['feature_bank_enabled'])
                if 'feature_bank_size' in config_data:
                    self.feature_bank_size = max(1, int(config_data['feature_bank_size']))
                if 'feature_bank_weight' in config_data:
                    self.feature_bank_weight = float(config_data['feature_bank_weight'])

                # Normalize prompt and seed configurations for frontend
                normalized_prompt_blending = self._normalize_prompt_config(config_data)
                normalized_seed_blending = self._normalize_seed_config(config_data)
                
                # Debug logging
                logger.debug(f"upload_controlnet_config: Raw prompt_blending in config: {config_data.get('prompt_blending', 'NOT FOUND')}")
                logger.debug(f"upload_controlnet_config: Raw seed_blending in config: {config_data.get('seed_blending', 'NOT FOUND')}")
                logger.debug(f"upload_controlnet_config: Normalized prompt blending: {normalized_prompt_blending}")
                logger.debug(f"upload_controlnet_config: Normalized seed blending: {normalized_seed_blending}")
                
                # Get other streaming parameters from config
                config_guidance_scale = config_data.get('guidance_scale', 1.1)
                config_delta = config_data.get('delta', 0.7)
                config_num_inference_steps = config_data.get('num_inference_steps', 50)
                config_seed = config_data.get('seed', 2)
                
                # Get normalization settings
                config_normalize_weights = config_data.get('normalize_weights', True)
                
                # Calculate current resolution string for frontend
                current_resolution = f"{self.new_width}x{self.new_height}"
                aspect_ratio = self._calculate_aspect_ratio(self.new_width, self.new_height)
                if aspect_ratio:
                    current_resolution += f" ({aspect_ratio})"
                
                # Get updated IPAdapter info for response
                response_ipadapter_info = self._get_ipadapter_info()

                
                return JSONResponse({
                    "status": "success",
                    "message": "ControlNet configuration uploaded successfully",
                    "controls_updated": True,  # Flag for frontend to update controls
                    "controlnet": self._get_controlnet_info(),
                    "ipadapter": response_ipadapter_info,  # Include updated IPAdapter info
                    "config_prompt": config_prompt,
                    "t_index_list": t_index_list,
                    "acceleration": config_acceleration,
                    "guidance_scale": config_guidance_scale,
                    "delta": config_delta,
                    "num_inference_steps": config_num_inference_steps,
                    "seed": config_seed,
                    "prompt_blending": normalized_prompt_blending,
                    "seed_blending": normalized_seed_blending,
                    "current_resolution": current_resolution,  # Include updated resolution
                    "normalize_prompt_weights": config_normalize_weights,
                    "normalize_seed_weights": config_normalize_weights,
                })
                
            except Exception as e:
                logging.error(f"upload_controlnet_config: Failed to upload config: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to upload configuration: {str(e)}")

        @self.app.get("/api/controlnet/info")
        async def get_controlnet_info():
            """Get current ControlNet configuration info"""
            return JSONResponse({"controlnet": self._get_controlnet_info()})

        @self.app.get("/api/blending/current")
        async def get_current_blending_config():
            """Get current prompt and seed blending configurations"""
            try:
                if self.pipeline and hasattr(self.pipeline, 'stream') and hasattr(self.pipeline.stream, 'get_stream_state'):
                    state = self.pipeline.stream.get_stream_state(include_caches=False)
                    return JSONResponse({
                        "prompt_blending": state.get("prompt_list", []),
                        "seed_blending": state.get("seed_list", []),
                        "normalize_prompt_weights": state.get("normalize_prompt_weights", True),
                        "normalize_seed_weights": state.get("normalize_seed_weights", True),
                        "has_config": self.uploaded_controlnet_config is not None,
                        "pipeline_active": True
                    })

                # Fallback to uploaded config normalization when pipeline not initialized
                prompt_blending_config = self._normalize_prompt_config(self.uploaded_controlnet_config)
                seed_blending_config = self._normalize_seed_config(self.uploaded_controlnet_config)
                normalize_weights = self.uploaded_controlnet_config.get('normalize_weights', True) if self.uploaded_controlnet_config else True
                return JSONResponse({
                    "prompt_blending": prompt_blending_config,
                    "seed_blending": seed_blending_config,
                    "normalize_prompt_weights": normalize_weights,
                    "normalize_seed_weights": normalize_weights,
                    "has_config": self.uploaded_controlnet_config is not None,
                    "pipeline_active": False
                })
                
            except Exception as e:
                logging.error(f"get_current_blending_config: Failed to get blending config: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get blending config: {str(e)}")

        @self.app.post("/api/controlnet/update-strength")
        async def update_controlnet_strength(request: Request):
            """Update ControlNet strength in real-time"""
            try:
                data = await request.json()
                controlnet_index = data.get("index")
                strength = data.get("strength")
                
                if controlnet_index is None or strength is None:
                    raise HTTPException(status_code=400, detail="Missing index or strength parameter")
                
                # Check if ControlNet is enabled using config system
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Check if we're using config mode and have controlnets configured
                controlnet_enabled = (self.pipeline.use_config and 
                                    self.pipeline.config and 
                                    'controlnets' in self.pipeline.config)
                
                if not controlnet_enabled:
                    raise HTTPException(status_code=400, detail="ControlNet is not enabled")
                
                # Update ControlNet strength using consolidated API
                current_config = self._get_current_controlnet_config()
                logger.info(f"update_controlnet_strength: Current config: {current_config}")
                
                if controlnet_index >= len(current_config):
                    raise HTTPException(status_code=400, detail=f"ControlNet index {controlnet_index} out of range")
                
                # Update only the conditioning_scale for the specified controlnet
                old_strength = current_config[controlnet_index]['conditioning_scale']
                current_config[controlnet_index]['conditioning_scale'] = float(strength)
                logger.info(f"update_controlnet_strength: Updating ControlNet {controlnet_index} strength from {old_strength} to {strength}")
                logger.info(f"update_controlnet_strength: Sending config: {current_config}")
                
                await self._async_pipeline_update(controlnet_config=current_config)
                logger.info(f"update_controlnet_strength: update_stream_params call completed")
                # Clear colour-match reference so the rolling EMA doesn't fight the new look
                self.color_match_reference = None

                return JSONResponse({
                    "status": "success",
                    "message": f"Updated ControlNet {controlnet_index} strength to {strength}"
                })
                
            except Exception as e:
                logging.error(f"update_controlnet_strength: Failed to update strength: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update strength: {str(e)}")

        @self.app.get("/api/controlnet/available")
        async def get_available_controlnets():
            """Get list of available ControlNets that can be added"""
            try:
                # Detect current model architecture to filter appropriate ControlNets
                model_type = "sd15"  # Default fallback
                
                if self.pipeline and hasattr(self.pipeline, 'config') and self.pipeline.config:
                    # Try to determine model type from config
                    model_id = self.pipeline.config.get('model_id', '')
                    if 'sdxl' in model_id.lower() or 'xl' in model_id.lower():
                        model_type = "sdxl"
                
                available = AVAILABLE_CONTROLNETS.get(model_type, [])
                
                # Filter out already active ControlNets
                current_controlnets = []
                # Check runtime config first, then fall back to uploaded config
                if self.runtime_controlnet_config and 'controlnets' in self.runtime_controlnet_config:
                    current_controlnets = [cn.get('model_id', '') for cn in self.runtime_controlnet_config['controlnets']]
                elif self.uploaded_controlnet_config and 'controlnets' in self.uploaded_controlnet_config:
                    current_controlnets = [cn.get('model_id', '') for cn in self.uploaded_controlnet_config['controlnets']]
                
                filtered_available = []
                for cn in available:
                    if cn['model_id'] not in current_controlnets:
                        filtered_available.append(cn)
                
                return JSONResponse({
                    "status": "success",
                    "available_controlnets": filtered_available,
                    "model_type": model_type
                })
                
            except Exception as e:
                logging.error(f"get_available_controlnets: Failed to get available ControlNets: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get available ControlNets: {str(e)}")

        @self.app.post("/api/controlnet/add")
        async def add_controlnet(request: Request):
            """Add a ControlNet from the predefined list"""
            try:
                data = await request.json()
                controlnet_id = data.get("controlnet_id")
                conditioning_scale = data.get("conditioning_scale", None)
                
                if not controlnet_id:
                    raise HTTPException(status_code=400, detail="Missing controlnet_id parameter")
                
                # Find the ControlNet definition
                controlnet_def = None
                for model_type, controlnets in AVAILABLE_CONTROLNETS.items():
                    for cn in controlnets:
                        if cn['id'] == controlnet_id:
                            controlnet_def = cn
                            break
                    if controlnet_def:
                        break
                
                if not controlnet_def:
                    raise HTTPException(status_code=400, detail=f"ControlNet {controlnet_id} not found in registry")
                
                # Use provided scale or default
                if conditioning_scale is None:
                    conditioning_scale = controlnet_def['default_scale']
                
                # Initialize runtime config from YAML if not already done
                if self.runtime_controlnet_config is None:
                    if self.uploaded_controlnet_config:
                        # Copy from YAML (deep copy to avoid modifying original)
                        import copy
                        self.runtime_controlnet_config = copy.deepcopy(self.uploaded_controlnet_config)
                    else:
                        # Create minimal config if no YAML exists
                        self.runtime_controlnet_config = {'controlnets': []}
                
                # Ensure controlnets key exists in runtime config
                if 'controlnets' not in self.runtime_controlnet_config:
                    self.runtime_controlnet_config['controlnets'] = []
                
                # Create new ControlNet entry
                new_controlnet = {
                    'model_id': controlnet_def['model_id'],
                    'conditioning_scale': conditioning_scale,
                    'preprocessor': controlnet_def['default_preprocessor'],
                    'preprocessor_params': controlnet_def.get('preprocessor_params', {}),
                    'enabled': True
                }
                
                # Add to runtime config (not YAML)
                self.runtime_controlnet_config['controlnets'].append(new_controlnet)
                
                # Update pipeline using consolidated API
                try:
                    current_config = self._get_current_controlnet_config()
                    current_config.append(new_controlnet)
                    await self._async_pipeline_update(controlnet_config=current_config)
                    logger.info(f"add_controlnet: Successfully added ControlNet using consolidated API")
                except Exception as e:
                    logger.error(f"add_controlnet: Failed to add ControlNet: {e}")
                    # Mark for reload as fallback
                    self.config_needs_reload = True
                
                logger.info(f"add_controlnet: Added {controlnet_def['name']} with scale {conditioning_scale}")
                
                # Return updated ControlNet info immediately
                updated_info = self._get_controlnet_info()
                added_index = len(self.runtime_controlnet_config['controlnets']) - 1
                
                return JSONResponse({
                    "status": "success", 
                    "message": f"Added {controlnet_def['name']}",
                    "controlnet_index": added_index,
                    "controlnet_info": updated_info
                })
                
            except Exception as e:
                logging.error(f"add_controlnet: Failed to add ControlNet: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to add ControlNet: {str(e)}")

        @self.app.get("/api/controlnet/status")
        async def get_controlnet_status():
            """Get the status of ControlNet configuration"""
            try:
                controlnet_pipeline = self._get_controlnet_pipeline()
                
                if not controlnet_pipeline:
                    return JSONResponse({
                        "status": "no_pipeline",
                        "message": "No ControlNet pipeline available",
                        "controlnet_count": 0
                    })
                
                current_config = self._get_current_controlnet_config()
                
                return JSONResponse({
                    "status": "ready",
                    "controlnet_count": len(current_config),
                    "message": f"{len(current_config)} ControlNet(s) configured" if current_config else "No ControlNets configured"
                })
                
            except Exception as e:
                logger.error(f"get_controlnet_status: Failed to get status: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

        @self.app.post("/api/controlnet/remove")
        async def remove_controlnet(request: Request):
            """Remove a ControlNet by index"""
            try:
                data = await request.json()
                index = data.get("index")
                
                if index is None:
                    raise HTTPException(status_code=400, detail="Missing index parameter")
                
                # Initialize runtime config from YAML if not already done
                if self.runtime_controlnet_config is None:
                    if self.uploaded_controlnet_config:
                        # Copy from YAML (deep copy to avoid modifying original)
                        import copy
                        self.runtime_controlnet_config = copy.deepcopy(self.uploaded_controlnet_config)
                    else:
                        raise HTTPException(status_code=400, detail="No ControlNet configuration found")
                
                if 'controlnets' not in self.runtime_controlnet_config:
                    raise HTTPException(status_code=400, detail="No ControlNet configuration found")
                
                controlnets = self.runtime_controlnet_config['controlnets']
                
                if index < 0 or index >= len(controlnets):
                    raise HTTPException(status_code=400, detail=f"ControlNet index {index} out of range")
                
                removed_controlnet = controlnets.pop(index)
                
                # Update pipeline using consolidated API
                try:
                    current_config = self._get_current_controlnet_config()
                    if index >= len(current_config):
                        raise HTTPException(status_code=400, detail=f"ControlNet index {index} out of range")
                    
                    # Remove the controlnet at the specified index
                    current_config.pop(index)
                    await self._async_pipeline_update(controlnet_config=current_config)
                    logger.info(f"remove_controlnet: Successfully removed ControlNet using consolidated API")
                except Exception as e:
                    logger.error(f"remove_controlnet: Failed to remove ControlNet: {e}")
                    # Mark for reload as fallback
                    self.config_needs_reload = True
                
                logger.info(f"remove_controlnet: Removed ControlNet at index {index}: {removed_controlnet.get('model_id', 'unknown')}")
                
                # Return updated ControlNet info immediately
                updated_info = self._get_controlnet_info()
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Removed ControlNet at index {index}",
                    "controlnet_info": updated_info
                })
                
            except Exception as e:
                logging.error(f"remove_controlnet: Failed to remove ControlNet: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to remove ControlNet: {str(e)}")

        @self.app.post("/api/ipadapter/upload-style-image")
        async def upload_style_image(file: UploadFile = File(...)):
            """Upload a style image for IPAdapter"""
            try:
                # Validate file type
                if not file.content_type or not file.content_type.startswith('image/'):
                    raise HTTPException(status_code=400, detail="File must be an image")
                
                # Read file content
                content = await file.read()
                
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        tmp.write(content)
                        tmp_path = tmp.name

                    # Load and validate image
                    from PIL import Image
                    style_image = Image.open(tmp_path).convert("RGB")

                    # Store the uploaded style image persistently FIRST
                    self.uploaded_style_image = style_image
                    print(f"upload_style_image: Stored style image with size: {style_image.size}")

                    # If pipeline exists and has IPAdapter, update it immediately
                    pipeline_updated = False
                    if self.pipeline and getattr(self.pipeline, 'has_ipadapter', False):
                        print("upload_style_image: Applying to existing pipeline")
                        success = self.pipeline.update_ipadapter_style_image(style_image)
                        if success:
                            pipeline_updated = True
                            print("upload_style_image: Successfully applied to existing pipeline")

                            # Force prompt re-encoding to apply new style image embeddings
                            try:
                                state = self.pipeline.stream.get_stream_state()
                                current_prompts = state.get('prompt_list', [])
                                if current_prompts:
                                    print("upload_style_image: Forcing prompt re-encoding to apply new style image")
                                    self.pipeline.stream.update_prompt(current_prompts, prompt_interpolation_method="slerp")
                                    print("upload_style_image: Prompt re-encoding completed")
                            except Exception as e:
                                print(f"upload_style_image: Failed to force prompt re-encoding: {e}")
                        else:
                            print("upload_style_image: Failed to apply to existing pipeline")
                    elif self.pipeline:
                        print(f"upload_style_image: Pipeline exists but has_ipadapter={getattr(self.pipeline, 'has_ipadapter', False)}")
                    else:
                        print("upload_style_image: No pipeline exists yet")

                    # Return success
                    message = "Style image uploaded successfully"
                    if pipeline_updated:
                        message += " and applied to active pipeline"
                    else:
                        message += " and will be applied when pipeline starts"
                    
                    return JSONResponse({
                        "status": "success",
                        "message": message
                    })
                finally:
                    if tmp_path:
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to upload style image: {str(e)}")

        @self.app.get("/api/ipadapter/uploaded-style-image")
        async def get_uploaded_style_image():
            """Get the currently uploaded style image"""
            try:
                if not self.uploaded_style_image:
                    raise HTTPException(status_code=404, detail="No style image uploaded")
                
                # Convert PIL image to bytes for streaming
                import io
                img_buffer = io.BytesIO()
                self.uploaded_style_image.save(img_buffer, format='JPEG', quality=95)
                img_buffer.seek(0)
                
                return StreamingResponse(
                    io.BytesIO(img_buffer.read()),
                    media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=3600"}
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to retrieve style image: {str(e)}")

        @self.app.get("/api/default-image")
        async def get_default_image():
            """Get the default image (input.png)"""
            try:
                import os
                default_image_path = os.path.join(os.path.dirname(__file__), "..", "..", "images", "inputs", "input.png")
                
                if not os.path.exists(default_image_path):
                    raise HTTPException(status_code=404, detail="Default image not found")
                
                # Read and return the default image file
                with open(default_image_path, "rb") as image_file:
                    image_content = image_file.read()
                
                return Response(content=image_content, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})
                
            except Exception as e:
                logging.error(f"get_default_image: Failed to retrieve default image: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to retrieve default image: {str(e)}")

        @self.app.post("/api/ipadapter/update-scale")
        async def update_ipadapter_scale(request: Request):
            """Update IPAdapter scale/strength in real-time"""
            try:
                data = await request.json()
                scale = data.get("scale")
                
                if scale is None:
                    raise HTTPException(status_code=400, detail="Missing scale parameter")
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Check if we're using config mode and have ipadapters configured
                ipadapter_enabled = (self.pipeline.use_config and 
                                    self.pipeline.config and 
                                    'ipadapters' in self.pipeline.config)
                
                if not ipadapter_enabled:
                    raise HTTPException(status_code=400, detail="IPAdapter is not enabled")
                
                # Update IPAdapter scale in the pipeline
                success = self.pipeline.update_ipadapter_scale(float(scale))

                if success:
                    # Clear colour-match reference so the rolling EMA doesn't fight the new look
                    self.color_match_reference = None
                    return JSONResponse({
                        "status": "success",
                        "message": f"Updated IPAdapter scale to {scale}"
                    })
                else:
                    raise HTTPException(status_code=500, detail="Failed to update scale in pipeline")
                
            except Exception as e:
                logging.error(f"update_ipadapter_scale: Failed to update scale: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update scale: {str(e)}")

        @self.app.post("/api/ipadapter/update-weight-type")
        async def update_ipadapter_weight_type(request: Request):
            """Update IPAdapter weight type in real-time"""
            try:
                data = await request.json()
                weight_type = data.get("weight_type")
                
                if weight_type is None:
                    raise HTTPException(status_code=400, detail="Missing weight_type parameter")
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Check if we're using config mode and have ipadapters configured
                ipadapter_enabled = (self.pipeline.use_config and 
                                    self.pipeline.config and 
                                    'ipadapters' in self.pipeline.config)
                
                if not ipadapter_enabled:
                    raise HTTPException(status_code=400, detail="IPAdapter is not enabled")
                
                # Update IPAdapter weight type in the pipeline
                success = self.pipeline.update_ipadapter_weight_type(weight_type)
                
                if success:
                    return JSONResponse({
                        "status": "success",
                        "message": f"Updated IPAdapter weight type to {weight_type}"
                    })
                else:
                    raise HTTPException(status_code=500, detail="Failed to update weight type in pipeline")
                
            except Exception as e:
                logging.error(f"update_ipadapter_weight_type: Failed to update weight type: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update weight type: {str(e)}")

        @self.app.post("/api/ipadapter/upload-style-image-b")
        async def upload_style_image_b(file: UploadFile = File(...)):
            """Upload blend slot B style image for IPAdapter"""
            try:
                if not file.content_type or not file.content_type.startswith('image/'):
                    raise HTTPException(status_code=400, detail="File must be an image")

                import tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    tmp_path = tmp.name
                try:
                    content = await file.read()
                    with open(tmp_path, 'wb') as f:
                        f.write(content)

                    from PIL import Image
                    style_image_b = Image.open(tmp_path).convert("RGB")
                    self.uploaded_style_image_b = style_image_b
                    print(f"upload_style_image_b: Stored image B size={style_image_b.size}")

                    pipeline_updated = False
                    if self.pipeline and getattr(self.pipeline, 'has_ipadapter', False):
                        success = self.pipeline.update_ipadapter_style_image_b(style_image_b)
                        print(f"upload_style_image_b: pipeline update success={success}")
                        if success:
                            pipeline_updated = True
                    else:
                        print(f"upload_style_image_b: no pipeline or no ipadapter")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

                return JSONResponse({
                    "status": "success",
                    "message": "Style image B uploaded successfully",
                    "pipeline_updated": pipeline_updated,
                })
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to upload style image B: {str(e)}")

        @self.app.get("/api/ipadapter/uploaded-style-image-b")
        async def get_uploaded_style_image_b():
            """Get the currently uploaded blend slot B style image"""
            try:
                if not self.uploaded_style_image_b:
                    raise HTTPException(status_code=404, detail="No style image B uploaded")

                import io
                img_buffer = io.BytesIO()
                self.uploaded_style_image_b.save(img_buffer, format='JPEG', quality=95)
                img_buffer.seek(0)
                return StreamingResponse(img_buffer, media_type="image/jpeg")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to retrieve style image B: {str(e)}")

        @self.app.get("/api/ipadapter/style-images-data")
        async def get_style_images_data():
            """Return current style images A and B as base64 JPEG for snapshot embedding."""
            import base64, io as _io
            MAX_SIZE = 512
            result = {}
            for key, attr in [('style_image_a', 'uploaded_style_image'), ('style_image_b', 'uploaded_style_image_b')]:
                img = getattr(self, attr, None)
                if img is not None:
                    try:
                        thumb = img.copy().convert('RGB')
                        thumb.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)
                        buf = _io.BytesIO()
                        thumb.save(buf, format='JPEG', quality=82)
                        result[key] = base64.b64encode(buf.getvalue()).decode('utf-8')
                    except Exception as e:
                        logger.warning(f"get_style_images_data: failed to encode {key}: {e}")
            return JSONResponse(result)

        @self.app.post("/api/ipadapter/blend-weight")
        async def update_ipadapter_blend_weight(request: Request):
            """Set IPAdapter blend weight between style image A (0.0) and B (1.0)"""
            try:
                data = await request.json()
                t = data.get("blend_weight")
                if t is None:
                    raise HTTPException(status_code=400, detail="blend_weight is required")

                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")

                if not getattr(self.pipeline, 'has_ipadapter', False):
                    raise HTTPException(status_code=400, detail="IPAdapter is not enabled")

                success = self.pipeline.update_ipadapter_blend_weight(float(t))
                if success:
                    return JSONResponse({"status": "success", "blend_weight": float(t)})
                else:
                    raise HTTPException(status_code=500, detail="Failed to update blend weight")
            except Exception as e:
                logging.error(f"update_ipadapter_blend_weight: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update blend weight: {str(e)}")

        @self.app.post("/api/params")
        async def update_params(request: Request):
            """Update multiple streaming parameters in a single unified call"""
            try:
                data = await request.json()
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Extract and validate parameters
                params = {}
                updated_params = []
                
                # Handle t_index_list
                if "t_index_list" in data:
                    t_index_list = data["t_index_list"]
                    if not isinstance(t_index_list, list) or not all(isinstance(x, int) for x in t_index_list):
                        raise HTTPException(status_code=400, detail="t_index_list must be a list of integers")
                    params["t_index_list"] = t_index_list
                    updated_params.append("t_index_list")
                
                # Handle guidance_scale
                if "guidance_scale" in data:
                    params["guidance_scale"] = float(data["guidance_scale"])
                    updated_params.append("guidance_scale")
                
                # Handle delta
                if "delta" in data:
                    params["delta"] = float(data["delta"])
                    updated_params.append("delta")
                
                # Handle num_inference_steps
                if "num_inference_steps" in data:
                    params["num_inference_steps"] = int(data["num_inference_steps"])
                    updated_params.append("num_inference_steps")
                
                # Handle seed
                if "seed" in data:
                    params["seed"] = int(data["seed"])
                    updated_params.append("seed")
                
                # Handle resolution (special case - triggers pipeline recreation)
                if "resolution" in data:
                    resolution = data["resolution"]
                    if isinstance(resolution, dict) and "width" in resolution and "height" in resolution:
                        width, height = int(resolution["width"]), int(resolution["height"])
                        self._update_resolution(width, height)
                        updated_params.append("resolution")
                    elif isinstance(resolution, str):
                        # Handle string format like "512x768 (2:3)"
                        resolution_part = resolution.split(' ')[0]  # Get "512x768" part
                        try:
                            width, height = map(int, resolution_part.split('x'))
                            self._update_resolution(width, height)
                            updated_params.append("resolution")
                        except ValueError:
                            raise HTTPException(status_code=400, detail="Invalid resolution format")
                    else:
                        raise HTTPException(status_code=400, detail="Resolution must be {width: int, height: int} or 'widthxheight' string")
                
                # Handle negative_prompt
                if "negative_prompt" in data:
                    neg = str(data["negative_prompt"])
                    params["negative_prompt"] = neg
                    if self.pipeline:
                        self.pipeline.negative_prompt = neg
                    updated_params.append("negative_prompt")

                # Handle normalization settings
                if "normalize_prompt_weights" in data:
                    params["normalize_prompt_weights"] = bool(data["normalize_prompt_weights"])
                    updated_params.append("normalize_prompt_weights")
                
                if "normalize_seed_weights" in data:
                    params["normalize_seed_weights"] = bool(data["normalize_seed_weights"])
                    updated_params.append("normalize_seed_weights")
                
                if not params and "resolution" not in data:
                    raise HTTPException(status_code=400, detail="No valid parameters provided")
                
                # Update parameters using unified API (excluding resolution which was handled above)
                if params:
                    await self._async_pipeline_update(**params)
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated parameters: {', '.join(updated_params)}",
                    "updated": updated_params
                })
                
            except HTTPException:
                raise
            except Exception as e:
                logging.error(f"update_params: Failed to update parameters: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update parameters: {str(e)}")

        @self.app.post("/api/upscale")
        async def set_upscale(request: Request):
            data = await request.json()
            factor = int(data.get("factor", 1))
            if factor not in (1, 2, 4):
                raise HTTPException(status_code=400, detail="factor must be 1, 2, or 4")
            self.upscale_factor = factor
            return JSONResponse({"status": "success", "upscale_factor": self.upscale_factor})

        @self.app.get("/api/upscale")
        async def get_upscale():
            return JSONResponse({"upscale_factor": self.upscale_factor})

        @self.app.post("/api/cfr")
        async def set_cfr(request: Request):
            """Enable/disable Constant Frame Rate pacing and set target FPS."""
            data = await request.json()
            if "enabled" in data:
                self.cfr_enabled = bool(data["enabled"])
            if "target_fps" in data:
                fps = int(data["target_fps"])
                if fps < 1 or fps > 120:
                    raise HTTPException(status_code=400, detail="target_fps must be between 1 and 120")
                self.target_fps = fps
            return JSONResponse({
                "status": "success",
                "cfr_enabled": self.cfr_enabled,
                "target_fps": self.target_fps,
            })

        @self.app.get("/api/cfr")
        async def get_cfr():
            """Return current CFR settings."""
            return JSONResponse({
                "cfr_enabled": self.cfr_enabled,
                "target_fps": self.target_fps,
            })

        @self.app.post("/api/color-match")
        async def set_color_match(request: Request):
            """Enable/disable color histogram match and set blend strength."""
            data = await request.json()
            if "enabled" in data:
                self.color_match_enabled = bool(data["enabled"])
            if "strength" in data:
                s = float(data["strength"])
                if not (0.0 <= s <= 1.0):
                    raise HTTPException(status_code=400, detail="strength must be 0.0–1.0")
                self.color_match_strength = s
            return JSONResponse({
                "status": "success",
                "color_match_enabled": self.color_match_enabled,
                "color_match_strength": self.color_match_strength,
                "has_reference": self.color_match_reference is not None,
            })

        @self.app.get("/api/color-match")
        async def get_color_match():
            """Return current color match settings."""
            return JSONResponse({
                "color_match_enabled": self.color_match_enabled,
                "color_match_strength": self.color_match_strength,
                "has_reference": self.color_match_reference is not None,
            })

        @self.app.post("/api/color-match/capture-reference")
        async def capture_color_reference():
            """Capture the current output frame as the color match reference."""
            if self.last_output_image is None:
                raise HTTPException(status_code=400, detail="No output frame available yet — start the stream first")
            try:
                src_rgb = np.array(self.last_output_image)
                self.color_match_reference = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR)
                return JSONResponse({"status": "success", "message": "Color reference captured"})
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to capture reference: {e}")

        @self.app.post("/api/color-match/clear-reference")
        async def clear_color_reference():
            """Clear the color match reference (disables matching until a new one is captured)."""
            self.color_match_reference = None
            return JSONResponse({"status": "success", "message": "Color reference cleared"})

        @self.app.post("/api/feature-bank")
        async def set_feature_bank(request: Request):
            """Enable/disable latent feature bank and set size and blend weight."""
            data = await request.json()
            if "enabled" in data:
                self.feature_bank_enabled = bool(data["enabled"])
            if "size" in data:
                s = int(data["size"])
                if not (1 <= s <= 8):
                    raise HTTPException(status_code=400, detail="size must be 1–8")
                self.feature_bank_size = s
            if "weight" in data:
                w = float(data["weight"])
                if not (0.0 <= w <= 1.0):
                    raise HTTPException(status_code=400, detail="weight must be 0.0–1.0")
                self.feature_bank_weight = w
            self._apply_feature_bank_settings()
            return JSONResponse({
                "status": "success",
                "feature_bank_enabled": self.feature_bank_enabled,
                "feature_bank_size": self.feature_bank_size,
                "feature_bank_weight": self.feature_bank_weight,
            })

        @self.app.get("/api/feature-bank")
        async def get_feature_bank():
            """Return current feature bank settings."""
            sd = self._get_inner_stream()
            bank_len = len(sd._feature_bank) if sd and hasattr(sd, '_feature_bank') else 0
            return JSONResponse({
                "feature_bank_enabled": self.feature_bank_enabled,
                "feature_bank_size": self.feature_bank_size,
                "feature_bank_weight": self.feature_bank_weight,
                "bank_frames_stored": bank_len,
            })

        @self.app.post("/api/feature-bank/reset")
        async def reset_feature_bank():
            """Clear the feature bank (useful after a large prompt or style change)."""
            sd = self._get_inner_stream()
            if sd and hasattr(sd, 'reset_feature_bank'):
                sd.reset_feature_bank()
            return JSONResponse({"status": "success", "message": "Feature bank cleared"})

        @self.app.get("/api/loras")
        async def get_loras():
            """Return available LoRA files from lora_dir and the current active list."""
            available = []
            try:
                if self.lora_dir.exists():
                    for f in sorted(self.lora_dir.glob("*.safetensors")):
                        available.append({
                            "name": f.stem,
                            "filename": f.name,
                            "path": str(f).replace("\\", "/"),
                        })
                    for f in sorted(self.lora_dir.glob("*.pt")):
                        available.append({
                            "name": f.stem,
                            "filename": f.name,
                            "path": str(f).replace("\\", "/"),
                        })
            except Exception as e:
                logger.error(f"get_loras: Failed to scan lora directory: {e}")

            active = (self.uploaded_controlnet_config or {}).get("loras", [])
            return JSONResponse({
                "available": available,
                "active": active,
                "lora_dir": str(self.lora_dir).replace("\\", "/"),
            })

        @self.app.post("/api/loras")
        async def update_loras(request: Request):
            """Update the active LoRA list. Triggers pipeline reload on next stream start."""
            data = await request.json()
            loras = data.get("loras", [])

            validated = []
            for entry in loras:
                if not isinstance(entry, dict) or "path" not in entry:
                    raise HTTPException(status_code=400, detail="Each LoRA entry must have a 'path' field")
                validated.append({
                    "path": str(entry["path"]),
                    "scale": float(entry.get("scale", 1.0)),
                    "enabled": bool(entry.get("enabled", True)),
                    "trigger_word": str(entry.get("trigger_word", "")).strip(),
                })

            if self.uploaded_controlnet_config is None:
                raise HTTPException(status_code=400, detail="No config loaded — load a YAML config first via the Load Config button")

            self.uploaded_controlnet_config["loras"] = validated
            if self.runtime_controlnet_config is not None:
                self.runtime_controlnet_config["loras"] = validated
            self.config_needs_reload = True

            # Collect trigger words from all enabled loras
            trigger_words = [e["trigger_word"] for e in validated if e["enabled"] and e["trigger_word"]]
            self.lora_trigger_words = ", ".join(trigger_words)
            logger.info(f"update_loras: Updated LoRA list ({len(validated)} entries), trigger_words={self.lora_trigger_words!r}, pipeline will reload")

            return JSONResponse({
                "status": "success",
                "message": f"LoRA list updated ({len(validated)} LoRA{'s' if len(validated) != 1 else ''}). Pipeline will restart on next stream start.",
                "loras": validated,
            })

        # Individual parameter update endpoints for input controls
        @self.app.post("/api/update-guidance-scale")
        async def update_guidance_scale(request: Request):
            """Update guidance scale parameter"""
            try:
                data = await request.json()
                guidance_scale = float(data.get("guidance_scale", 1.0))
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                await self._async_pipeline_update(guidance_scale=guidance_scale)
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated guidance_scale to {guidance_scale}",
                    "guidance_scale": guidance_scale
                })
                
            except Exception as e:
                logging.error(f"update_guidance_scale: Failed to update guidance scale: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update guidance scale: {str(e)}")

        @self.app.post("/api/update-delta")
        async def update_delta(request: Request):
            """Update delta parameter"""
            try:
                data = await request.json()
                delta = float(data.get("delta", 0.7))
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                await self._async_pipeline_update(delta=delta)
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated delta to {delta}",
                    "delta": delta
                })
                
            except Exception as e:
                logging.error(f"update_delta: Failed to update delta: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update delta: {str(e)}")

        @self.app.post("/api/update-num-inference-steps")
        async def update_num_inference_steps(request: Request):
            """Update number of inference steps parameter"""
            try:
                data = await request.json()
                num_inference_steps = int(data.get("num_inference_steps", 50))
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                await self._async_pipeline_update(num_inference_steps=num_inference_steps)
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated num_inference_steps to {num_inference_steps}",
                    "num_inference_steps": num_inference_steps
                })
                
            except Exception as e:
                logging.error(f"update_num_inference_steps: Failed to update num_inference_steps: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update num_inference_steps: {str(e)}")

        @self.app.post("/api/update-seed")
        async def update_seed(request: Request):
            """Update seed parameter"""
            try:
                data = await request.json()
                seed = int(data.get("seed", 2))
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                await self._async_pipeline_update(seed=seed)
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated seed to {seed}",
                    "seed": seed
                })
                
            except Exception as e:
                logging.error(f"update_seed: Failed to update seed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update seed: {str(e)}")

        @self.app.post("/api/blending")
        async def update_blending(request: Request):
            """Update prompt and/or seed blending configuration in real-time"""
            try:
                data = await request.json()
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                params = {}
                updated_types = []
                
                # Handle prompt blending
                if "prompt_list" in data:
                    prompt_list = data["prompt_list"]
                    interpolation_method = data.get("prompt_interpolation_method", "slerp")
                    
                    if not isinstance(prompt_list, list):
                        raise HTTPException(status_code=400, detail="prompt_list must be a list")
                    
                    # Validate and convert format
                    prompt_tuples = []
                    for item in prompt_list:
                        if isinstance(item, list) and len(item) == 2:
                            prompt_tuples.append((str(item[0]), float(item[1])))
                        elif isinstance(item, dict) and "prompt" in item and "weight" in item:
                            prompt_tuples.append((str(item["prompt"]), float(item["weight"])))
                        else:
                            raise HTTPException(status_code=400, detail="Each prompt item must be [prompt, weight] or {prompt: str, weight: float}")
                    
                    params["prompt_list"] = prompt_tuples
                    params["prompt_interpolation_method"] = interpolation_method
                    updated_types.append("prompt blending")
                
                # Handle seed blending
                if "seed_list" in data:
                    seed_list = data["seed_list"]
                    interpolation_method = data.get("seed_interpolation_method", "linear")
                    
                    if not isinstance(seed_list, list):
                        raise HTTPException(status_code=400, detail="seed_list must be a list")
                    
                    # Validate and convert format
                    seed_tuples = []
                    for item in seed_list:
                        if isinstance(item, list) and len(item) == 2:
                            seed_tuples.append((int(item[0]), float(item[1])))
                        elif isinstance(item, dict) and "seed" in item and "weight" in item:
                            seed_tuples.append((int(item["seed"]), float(item["weight"])))
                        else:
                            raise HTTPException(status_code=400, detail="Each seed item must be [seed, weight] or {seed: int, weight: float}")
                    
                    params["seed_list"] = seed_tuples
                    params["seed_interpolation_method"] = interpolation_method
                    updated_types.append("seed blending")
                
                if not params:
                    raise HTTPException(status_code=400, detail="No blending parameters provided")
                
                # Update blending using unified API
                await self._async_pipeline_update(**params)
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Updated {' and '.join(updated_types)}",
                    "updated": updated_types
                })
                
            except HTTPException:
                raise
            except Exception as e:
                logging.error(f"update_blending: Failed to update blending: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update blending: {str(e)}")

        @self.app.get("/api/fps")
        async def get_fps():
            """Get current FPS plus per-phase timing and CUDA memory diagnostics."""
            if len(self.fps_counter) > 0:
                avg_frame_time = sum(self.fps_counter) / len(self.fps_counter)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            else:
                fps = 0

            result = {"fps": round(fps, 1)}

            # Per-phase timing averages
            if self.timing_wait_ms:
                result["timing_ms"] = {
                    "wait":    round(sum(self.timing_wait_ms)    / len(self.timing_wait_ms),    1),
                    "predict": round(sum(self.timing_predict_ms) / len(self.timing_predict_ms), 1),
                    "encode":  round(sum(self.timing_encode_ms)  / len(self.timing_encode_ms),  1),
                }

            # CUDA memory
            if torch.cuda.is_available():
                result["cuda_mb"] = {
                    "allocated": round(torch.cuda.memory_allocated() / 1e6, 1),
                    "reserved":  round(torch.cuda.memory_reserved()  / 1e6, 1),
                    "free_in_reserved": round(
                        (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1e6, 1
                    ),
                }

            return JSONResponse(result)

        @self.app.post("/api/save_snapshot")
        async def save_snapshot(request: Request):
            """Save the latest output frame as a PNG with embedded parameter metadata."""
            try:
                body = {}
                try:
                    body = await request.json()
                except Exception:
                    pass
                name = (body.get('name') or '').strip()
                # Prefer params sent from the frontend (authoritative live UI state)
                frontend_params = body.get('params')
            except Exception:
                name = ''
                frontend_params = None

            snapshots_dir = Path(__file__).parent / "snapshots"
            snapshots_dir.mkdir(parents=True, exist_ok=True)

            if self.last_output_image is None:
                logger.warning("save_snapshot: last_output_image is None — stream may not have produced a frame yet")
                raise HTTPException(status_code=400, detail="No output frame yet — the stream must be running and have produced at least one frame")

            filename = f"{name}.png" if name else datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
            save_path = snapshots_dir / filename

            if frontend_params:
                # Frontend sent current UI state — use it directly (most accurate)
                params = frontend_params
                logger.info("save_snapshot: using frontend-provided params")
            else:
                # Fallback: read from pipeline (less reliable for some params)
                params = self._collect_snapshot_params()
                logger.info("save_snapshot: using backend-collected params (fallback)")
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("sdiff_params", json.dumps(params))

            self.last_output_image.save(str(save_path), format="PNG", pnginfo=pnginfo)
            logger.info(f"save_snapshot: Saved to {save_path}")

            return JSONResponse({"status": "success", "path": str(save_path), "filename": filename})

        @self.app.post("/api/load_snapshot")
        async def load_snapshot(file: UploadFile = File(...)):
            """Read sdiff_params metadata from a saved snapshot PNG and return the params."""
            try:
                import io as _io
                content = await file.read()
                img = Image.open(_io.BytesIO(content))
                params_json = img.info.get("sdiff_params")
                if not params_json:
                    raise HTTPException(status_code=400, detail="No sdiff_params metadata found in this image")
                params = json.loads(params_json)
                logger.info(f"load_snapshot: Loaded params from {file.filename}")
                return JSONResponse({"status": "success", "params": params})
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"load_snapshot: Failed to read snapshot: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to read snapshot: {str(e)}")

        @self.app.get("/api/preprocessors/info")
        async def get_preprocessors_info():
            """Get preprocessor information using metadata from preprocessor classes"""
            try:
                from streamdiffusion.preprocessing.processors import list_preprocessors, get_preprocessor
                
                available_preprocessors = list_preprocessors()
                preprocessors_info = {}
                
                for preprocessor_name in available_preprocessors:
                    try:
                        preprocessor_class = get_preprocessor(preprocessor_name).__class__
                        
                        # Get comprehensive metadata from class
                        metadata = preprocessor_class.get_preprocessor_metadata()
                        
                        # Use metadata directly, with the preprocessor name as key
                        preprocessors_info[preprocessor_name] = metadata
                        
                    except Exception as e:
                        logger.warning(f"get_preprocessors_info: Could not extract info for {preprocessor_name}: {e}")
                        # Fallback to basic info if metadata method fails
                        preprocessors_info[preprocessor_name] = {
                            "display_name": preprocessor_name.replace("_", " ").title(),
                            "description": f"Preprocessor for {preprocessor_name}",
                            "parameters": {},
                            "use_cases": []
                        }
                        continue
                
                return JSONResponse({
                    "preprocessors": preprocessors_info,
                    "available": available_preprocessors
                })
                
            except Exception as e:
                logger.error(f"get_preprocessors_info: Error loading preprocessor info: {e}")
                return JSONResponse({
                    "preprocessors": {},
                    "available": [],
                    "error": "Could not load preprocessor information"
                })

        @self.app.post("/api/preprocessors/switch")
        async def switch_preprocessor(request: Request):
            """Switch preprocessor for a specific ControlNet"""
            try:
                data = await request.json()
                controlnet_index = data.get("controlnet_index", 0)
                new_preprocessor = data.get("preprocessor")
                preprocessor_params = data.get("preprocessor_params", {})
                
                logger.info(f"switch_preprocessor: Switching ControlNet {controlnet_index} to {new_preprocessor}")
                
                if not new_preprocessor:
                    raise HTTPException(status_code=400, detail="Missing preprocessor parameter")
                
                # Get ControlNet pipeline using helper
                cn_pipeline = self._get_controlnet_pipeline()
                if not cn_pipeline:
                    raise HTTPException(status_code=400, detail="ControlNet pipeline not found")
                
                if controlnet_index >= len(cn_pipeline.preprocessors):
                    raise HTTPException(status_code=400, detail=f"ControlNet index {controlnet_index} out of range")
                
                # Create new preprocessor instance
                from streamdiffusion.preprocessing.processors import get_preprocessor
                new_preprocessor_instance = get_preprocessor(new_preprocessor)

                # Resolve stream object and preprocessor list regardless of module or stream facade
                stream_obj = getattr(cn_pipeline, '_stream', None)
                if stream_obj is None:
                    stream_obj = getattr(self.pipeline, 'stream', None)
                if stream_obj is None:
                    raise HTTPException(status_code=500, detail="Pipeline stream not available")

                preproc_list = getattr(cn_pipeline, 'preprocessors', None)
                if preproc_list is None:
                    preproc_list = getattr(stream_obj, 'preprocessors', None)
                if preproc_list is None:
                    raise HTTPException(status_code=500, detail="ControlNet preprocessors not available")

                # Set system parameters
                system_params = {
                    'device': stream_obj.device,
                    'dtype': stream_obj.dtype,
                    'image_width': stream_obj.width,
                    'image_height': stream_obj.height,
                }
                system_params.update(preprocessor_params)
                new_preprocessor_instance.params.update(system_params)

                # Set pipeline reference for feedback preprocessor
                if hasattr(new_preprocessor_instance, 'set_pipeline_ref'):
                    new_preprocessor_instance.set_pipeline_ref(stream_obj)

                # Replace the preprocessor
                old_preprocessor = preproc_list[controlnet_index]
                preproc_list[controlnet_index] = new_preprocessor_instance
                
                logger.info(f"switch_preprocessor: Successfully switched ControlNet {controlnet_index} from {type(old_preprocessor).__name__ if old_preprocessor else 'None'} to {type(new_preprocessor_instance).__name__}")
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Successfully switched to {new_preprocessor} preprocessor",
                    "controlnet_index": controlnet_index,
                    "preprocessor": new_preprocessor,
                    "parameters": preprocessor_params
                })
                    
            except Exception as e:
                logger.error(f"switch_preprocessor: Failed to switch preprocessor: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to switch preprocessor: {str(e)}")
        
        @self.app.post("/api/preprocessors/update-params")
        async def update_preprocessor_params(request: Request):
            """Update preprocessor parameters for a specific ControlNet"""
            try:
                data = await request.json()
                controlnet_index = data.get("controlnet_index", 0)
                preprocessor_params = data.get("preprocessor_params", {})
                

                
                if not preprocessor_params:
                    raise HTTPException(status_code=400, detail="Missing preprocessor_params parameter")
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Fast path: update module preprocessor directly when available
                cn_pipeline = self._get_controlnet_pipeline()
                preproc_list = getattr(cn_pipeline, 'preprocessors', None)
                if preproc_list is None:
                    raise HTTPException(status_code=400, detail="ControlNet preprocessors not available")

                if controlnet_index >= len(preproc_list):
                    raise HTTPException(status_code=400, detail=f"ControlNet index {controlnet_index} out of range (max: {len(preproc_list)-1})")

                target_preproc = preproc_list[controlnet_index]
                if target_preproc is None:
                    raise HTTPException(status_code=400, detail="ControlNet preprocessor is not set")

                # Merge params: update both the params map and setattr when attribute exists
                if hasattr(target_preproc, 'params') and isinstance(target_preproc.params, dict):
                    target_preproc.params.update(preprocessor_params)
                for name, value in preprocessor_params.items():
                    if hasattr(target_preproc, name):
                        setattr(target_preproc, name, value)

                # Clear colour-match reference so the rolling EMA adapts to the new look
                self.color_match_reference = None

                return JSONResponse({
                    "status": "success",
                    "message": "Successfully updated preprocessor parameters",
                    "controlnet_index": controlnet_index,
                    "updated_parameters": preprocessor_params
                })
                    
            except Exception as e:
                logger.error(f"update_preprocessor_params: Failed to update parameters: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update preprocessor parameters: {str(e)}")

        @self.app.post("/api/blending/update-prompt-weight")
        async def update_prompt_weight(request: Request):
            """Update a specific prompt weight in the current blending configuration"""
            try:
                data = await request.json()
                index = data.get('index')
                weight = data.get('weight')
                
                if index is None or weight is None:
                    raise HTTPException(status_code=400, detail="Missing index or weight parameter")
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Get current prompt blending configuration via unified getter, fallback to uploaded config
                state = self.pipeline.stream.get_stream_state()
                current_prompts = state.get('prompt_list') or self._normalize_prompt_config(self.uploaded_controlnet_config)
                    
                if current_prompts and index < len(current_prompts):
                    # Create updated prompt list with new weight
                    updated_prompts = list(current_prompts)  # Make a copy
                    updated_prompts[index] = (updated_prompts[index][0], float(weight))
                    
                    # Use the same update method as the main blending endpoint
                    params = {
                        "prompt_list": updated_prompts,
                        "prompt_interpolation_method": "slerp"  # Default method
                    }
                    
                    # Apply the update using the working method
                    await self._async_pipeline_update(**params)

                    return JSONResponse({
                        "status": "success",
                        "message": f"Successfully updated prompt {index} weight",
                        "index": index,
                        "weight": weight
                    })
                else:
                    raise HTTPException(status_code=400, detail=f"Prompt index {index} out of range or no prompts available")
                    
            except Exception as e:
                logger.error(f"update_prompt_weight: Failed to update prompt weight: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update prompt weight: {str(e)}")

        @self.app.post("/api/blending/update-seed-weight") 
        async def update_seed_weight(request: Request):
            """Update a specific seed weight in the current blending configuration"""
            try:
                data = await request.json()
                index = data.get('index')
                weight = data.get('weight')
                
                if index is None or weight is None:
                    raise HTTPException(status_code=400, detail="Missing index or weight parameter")
                
                if not self.pipeline:
                    raise HTTPException(status_code=400, detail="Pipeline is not initialized")
                
                # Get current seed blending configuration via unified getter, fallback to uploaded config
                state = self.pipeline.stream.get_stream_state()
                current_seeds = state.get('seed_list') or self._normalize_seed_config(self.uploaded_controlnet_config)
                    
                if current_seeds and index < len(current_seeds):
                    # Create updated seed list with new weight
                    updated_seeds = list(current_seeds)  # Make a copy
                    updated_seeds[index] = (updated_seeds[index][0], float(weight))
                    
                    # Use the same update method as the main blending endpoint
                    params = {
                        "seed_list": updated_seeds,
                        "seed_interpolation_method": "linear"  # Default method
                    }
                    
                    # Apply the update using the working method
                    await self._async_pipeline_update(**params)

                    return JSONResponse({
                        "status": "success",
                        "message": f"Successfully updated seed {index} weight",
                        "index": index,
                        "weight": weight
                    })
                else:
                    raise HTTPException(status_code=400, detail=f"Seed index {index} out of range or no seeds available")
                    
            except Exception as e:
                logger.error(f"update_seed_weight: Failed to update seed weight: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update seed weight: {str(e)}")

        @self.app.get("/api/preprocessors/current-params/{controlnet_index}")
        async def get_current_preprocessor_params(controlnet_index: int):
            """Get current parameter values for a specific ControlNet preprocessor"""
            try:
                # Quick fix: return empty params if pipeline not ready
                if not self.pipeline:
                    return JSONResponse({
                        "preprocessor": None,
                        "parameters": {}
                    })
                
                # Get ControlNet pipeline using helper
                cn_pipeline = self._get_controlnet_pipeline()
                if not cn_pipeline:
                    return JSONResponse({
                        "preprocessor": None, 
                        "parameters": {}
                    })
                
                # Module-aware: allow accessing module's preprocessors list
                preprocessors = getattr(cn_pipeline, 'preprocessors', None)
                if preprocessors is None:
                    return JSONResponse({
                        "preprocessor": None,
                        "parameters": {}
                    })
                if controlnet_index >= len(preprocessors):
                    return JSONResponse({
                        "preprocessor": None,
                        "parameters": {}
                    })
                
                current_preprocessor = preprocessors[controlnet_index]
                if not current_preprocessor:
                    return JSONResponse({
                        "preprocessor": None,
                        "parameters": {}
                    })
                
                # Get user-configurable parameters metadata
                metadata = current_preprocessor.__class__.get_preprocessor_metadata()
                user_param_meta = metadata.get("parameters", {})
                
                # Extract current values, using defaults if not set
                current_values = {}
                for param_name, param_meta in user_param_meta.items():
                    if hasattr(current_preprocessor, 'params') and param_name in current_preprocessor.params:
                        current_values[param_name] = current_preprocessor.params[param_name]
                    else:
                        current_values[param_name] = param_meta.get("default")
                
                return JSONResponse({
                    "preprocessor": current_preprocessor.__class__.__name__.replace("Preprocessor", "").lower(),
                    "parameters": current_values
                })
                    
            except Exception as e:
                logger.error(f"get_current_preprocessor_params: Failed to get current parameters: {e}")
                return JSONResponse({
                    "preprocessor": None,
                    "parameters": {}
                })
                
                # Get user-configurable parameters metadata
                metadata = current_preprocessor.__class__.get_preprocessor_metadata()
                user_param_meta = metadata.get("parameters", {})
                
                # Extract current values, using defaults if not set
                current_values = {}
                for param_name, param_meta in user_param_meta.items():
                    if hasattr(current_preprocessor, 'params') and param_name in current_preprocessor.params:
                        current_values[param_name] = current_preprocessor.params[param_name]
                    else:
                        current_values[param_name] = param_meta.get("default")
                
                return JSONResponse({
                    "preprocessor": current_preprocessor.__class__.__name__.replace("Preprocessor", "").lower(),
                    "parameters": current_values
                })
                    
            except Exception as e:
                logger.error(f"get_current_preprocessor_params: Failed to get current parameters: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get current preprocessor parameters: {str(e)}")

        

        
        # Only mount static files if not in API-only mode
        
         # ADD THE VIDEO ENDPOINTS HERE (right after the last existing endpoint):

        @self.app.post("/api/video-input/upload")
        async def upload_video_file(file: UploadFile = File(...)):
            """Upload and set a video file for input"""
            try:
                if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    raise HTTPException(status_code=400, detail="File must be a video file")

                # --- START of CHANGES ---

                # If a video is currently playing, stop it before loading the new one.
                if self.video_input_active:
                    self.video_input_manager.stop()
                    self.video_input_active = False # Temporarily set to false

                # Save uploaded file temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    content = await file.read()
                    tmp.write(content)
                    tmp_path = tmp.name
                
                # Load the video
                success = self.video_input_manager.load_video(tmp_path)
                
                if success:
                    self.video_input_mode = "video_file"
                    video_info = self.video_input_manager.get_video_info()

                    # Automatically start playback of the new video
                    self.video_input_manager.start_playback(loop=True) # Assume loop=True is a good default
                    self.video_input_active = True
                    
                    return JSONResponse({
                        "status": "success",
                        "message": f"Video uploaded and started: {file.filename}",
                        "video_info": video_info
                    })
                else:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                    raise HTTPException(status_code=400, detail="Failed to load video file")

                # --- END of CHANGES ---
                
            except Exception as e:
                logging.error(f"upload_video_file: Failed to upload video: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to upload video: {str(e)}")

        @self.app.get("/api/video-input/status")
        async def get_video_input_status():
            """Get current video input status"""
            try:
                status = {
                    "active": self.video_input_active,
                    "mode": self.video_input_mode,
                    "video_info": self.video_input_manager.get_video_info() if self.video_input_mode == "video_file" else None
                }
                return JSONResponse(status)
            except Exception as e:
                logging.error(f"get_video_input_status: Failed to get status: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get video input status: {str(e)}")

        @self.app.post("/api/video-input/start")
        async def start_video_input(request: Request):
            """Start video input, respecting the current mode and avoiding restarts if already active."""
            try:
                data = await request.json()
                input_mode = data.get("mode", self.video_input_mode)
                loop_video = data.get("loop", True)
                
                if input_mode == "video_file":
                    # --- START of FIX ---
                    # If the video is already active in this mode, do nothing.
                    if self.video_input_active and self.video_input_mode == "video_file":
                        logging.info("start_video_input: Video file input is already active. No action taken.")
                        return JSONResponse({
                            "status": "success",
                            "message": "Video file input was already active",
                            "mode": "video_file"
                        })
                    # --- END of FIX ---

                    if not self.video_input_manager.video_path:
                        raise HTTPException(status_code=400, detail="No video file loaded")
                    
                    self.video_input_manager.stop() # This is safe now, as we're only here if it's inactive.
                    success = self.video_input_manager.start_playback(loop=loop_video)
                    
                    if success:
                        self.video_input_active = True
                        self.video_input_mode = "video_file"
                        return JSONResponse({
                            "status": "success",
                            "message": "Video file input started",
                            "mode": "video_file"
                        })
                    else:
                        raise HTTPException(status_code=500, detail="Failed to start video playback")
                else:
                    # Logic for switching to webcam
                    self.video_input_manager.stop()
                    self.video_input_mode = "webcam"
                    self.video_input_active = True
                    return JSONResponse({
                        "status": "success", 
                        "message": "Webcam input mode set",
                        "mode": "webcam"
                    })
                
            except Exception as e:
                logging.error(f"start_video_input: Failed to start video input: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to start video input: {str(e)}")

        @self.app.post("/api/video-input/stop")
        async def stop_video_input():
            """Stop video input"""
            try:
                self.video_input_manager.stop()
                self.video_input_active = False
                return JSONResponse({"status": "success", "message": "Video input stopped"})
            except Exception as e:
                logging.error(f"stop_video_input: Failed to stop video input: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to stop video input: {str(e)}")

        @self.app.get("/api/video-input/frame")
        async def get_current_video_frame():
            """Get the current video frame as JPEG"""
            try:
                if self.video_input_mode == "video_file":
                    frame = self.video_input_manager.get_current_frame()
                    if frame is not None:
                        frame_bytes = self.video_input_manager.frame_to_bytes(frame)
                        return Response(content=frame_bytes, media_type="image/jpeg")
                    else:
                        raise HTTPException(status_code=404, detail="No frame available")
                else:
                    raise HTTPException(status_code=400, detail="Not in video file mode")
            except Exception as e:
                logging.error(f"get_current_video_frame: Failed to get frame: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get current frame: {str(e)}")       
        
    
      
        
        # Input control management endpoints
        @self.app.post("/api/input-control/add")
        async def add_input_control(request: Request):
            """Add a new input control"""
            try:
                data = await request.json()
                
                input_id = data.get("input_id")
                input_type = data.get("input_type")
                parameter_name = data.get("parameter_name")
                min_value = data.get("min_value", 0.0)
                max_value = data.get("max_value", 1.0)
                
                if not all([input_id, input_type, parameter_name]):
                    raise HTTPException(status_code=400, detail="Missing required parameters: input_id, input_type, parameter_name")
                
                # Handle different input types
                if input_type == "gamepad":
                    # Backend gamepad control
                    gamepad_index = data.get("gamepad_index", 0)
                    axis_index = data.get("axis_index", 0)
                    deadzone = data.get("deadzone", 0.1)
                    
                    gamepad_control = GamepadInput(
                        parameter_name=parameter_name,
                        min_value=min_value,
                        max_value=max_value,
                        gamepad_index=gamepad_index,
                        axis_index=axis_index,
                        deadzone=deadzone
                    )
                    
                    self.input_manager.add_input(input_id, gamepad_control)
                    logger.info(f"add_input_control: Added gamepad control for parameter {parameter_name}")
                    
                elif input_type == "microphone":
                    # Frontend-based control
                    raise HTTPException(status_code=400, detail="Microphone inputs are managed in the frontend")
                elif input_type == "hand_tracking":
                    # Frontend-based control
                    raise HTTPException(status_code=400, detail="Hand tracking inputs are managed in the frontend")
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported input type: {input_type}")
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Added {input_type} input control for {parameter_name}"
                })
                
            except Exception as e:
                logging.error(f"add_input_control: Failed to add input control: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to add input control: {str(e)}")

        @self.app.post("/api/input-control/start/{input_id}")
        async def start_input_control(input_id: str):
            """Start a specific input control"""
            try:
                await self.input_manager.start_input(input_id)
                return JSONResponse({
                    "status": "success",
                    "message": f"Started input control {input_id}"
                })
            except Exception as e:
                logging.error(f"start_input_control: Failed to start input control {input_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to start input control: {str(e)}")

        @self.app.post("/api/input-control/stop/{input_id}")
        async def stop_input_control(input_id: str):
            """Stop a specific input control"""
            try:
                await self.input_manager.stop_input(input_id)
                return JSONResponse({
                    "status": "success",
                    "message": f"Stopped input control {input_id}"
                })
            except Exception as e:
                logging.error(f"stop_input_control: Failed to stop input control {input_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to stop input control: {str(e)}")

        @self.app.delete("/api/input-control/{input_id}")
        async def remove_input_control(input_id: str):
            """Remove an input control"""
            try:
                self.input_manager.remove_input(input_id)
                return JSONResponse({
                    "status": "success",
                    "message": f"Removed input control {input_id}"
                })
            except Exception as e:
                logging.error(f"remove_input_control: Failed to remove input control {input_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to remove input control: {str(e)}")

        @self.app.get("/api/input-control/status")
        async def get_input_control_status():
            """Get status of all input controls"""
            try:
                status = self.input_manager.get_input_status()
                return JSONResponse({
                    "status": "success",
                    "input_controls": status
                })
            except Exception as e:
                logging.error(f"get_input_control_status: Failed to get input control status: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get input control status: {str(e)}")

        # ── VirtCam endpoints ────────────────────────────────────────────────
        @self.app.get("/api/virtcam/videos")
        async def list_virtcam_videos():
            """List available video files from the source video directory."""
            VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
            files = []
            if self._source_video_dir.exists():
                for f in sorted(self._source_video_dir.iterdir()):
                    if f.suffix.lower() in VIDEO_EXTS:
                        files.append({"name": f.name, "path": str(f)})
            return JSONResponse({"videos": files, "directory": str(self._source_video_dir)})

        @self.app.get("/api/virtcam/status")
        async def virtcam_status():
            """Return whether virtcam is running and which file."""
            running = self.virtcam_process is not None and self.virtcam_process.poll() is None
            return JSONResponse({
                "running": running,
                "video_path": self.virtcam_video_path if running else None,
                "video_name": Path(self.virtcam_video_path).name if running and self.virtcam_video_path else None,
                "speed": self.virtcam_speed,
            })

        @self.app.post("/api/virtcam/start")
        async def start_virtcam(request: Request):
            """Start virtcam.py with the given video file path."""
            body = await request.json()
            video_path = body.get("video_path", "").strip()
            speed = max(1, min(5, int(body.get("speed", self.virtcam_speed))))
            if not video_path:
                raise HTTPException(status_code=400, detail="video_path is required")
            if not Path(video_path).exists():
                raise HTTPException(status_code=400, detail=f"File not found: {video_path}")
            if not self._virtcam_script.exists():
                raise HTTPException(status_code=500, detail=f"virtCam.py not found at {self._virtcam_script}")

            # Stop existing process if running
            if self.virtcam_process and self.virtcam_process.poll() is None:
                self.virtcam_process.terminate()
                try:
                    self.virtcam_process.wait(timeout=3)
                except Exception:
                    self.virtcam_process.kill()

            import subprocess as _sp
            self.virtcam_speed = speed
            self.virtcam_process = _sp.Popen(
                [sys.executable, str(self._virtcam_script), video_path, "--speed", str(speed)],
                stdout=_sp.DEVNULL,
                stderr=_sp.DEVNULL,
            )
            self.virtcam_video_path = video_path
            logger.info(f"virtcam started: {video_path} speed={speed} (pid {self.virtcam_process.pid})")
            return JSONResponse({"status": "started", "video_path": video_path, "speed": speed, "pid": self.virtcam_process.pid})

        @self.app.post("/api/virtcam/speed")
        async def set_virtcam_speed(request: Request):
            """Change playback speed while running; restarts the subprocess."""
            body = await request.json()
            speed = max(1, min(5, int(body.get("speed", 1))))
            self.virtcam_speed = speed

            if self.virtcam_process and self.virtcam_process.poll() is None and self.virtcam_video_path:
                import subprocess as _sp
                self.virtcam_process.terminate()
                try:
                    self.virtcam_process.wait(timeout=3)
                except Exception:
                    self.virtcam_process.kill()
                self.virtcam_process = _sp.Popen(
                    [sys.executable, str(self._virtcam_script), self.virtcam_video_path, "--speed", str(speed)],
                    stdout=_sp.DEVNULL,
                    stderr=_sp.DEVNULL,
                )
                logger.info(f"virtcam speed changed to {speed} (pid {self.virtcam_process.pid})")

            return JSONResponse({"status": "ok", "speed": speed})

        @self.app.post("/api/virtcam/stop")
        async def stop_virtcam():
            """Stop the running virtcam process."""
            if self.virtcam_process and self.virtcam_process.poll() is None:
                self.virtcam_process.terminate()
                try:
                    self.virtcam_process.wait(timeout=3)
                except Exception:
                    self.virtcam_process.kill()
                logger.info("virtcam stopped")
            self.virtcam_process = None
            self.virtcam_video_path = None
            return JSONResponse({"status": "stopped"})

        # Static files mount MUST be last — it acts as a catch-all and shadows any routes registered after it
        if not self.args.api_only:
            if not os.path.exists("public"):
                os.makedirs("public")
            self.app.mount(
                "/", StaticFiles(directory="./frontend/public", html=True), name="public"
            )
        else:
            @self.app.get("/")
            async def api_root():
                return JSONResponse({
                    "message": "StreamDiffusion API Server",
                    "mode": "api-only",
                    "frontend": "Run separately with 'npm run dev' in ./frontend/"
                })

    def _normalize_prompt_config(self, config_data):
        """
        Normalize prompt configuration to always return a list format.
        Priority: prompt_blending.prompt_list > prompt_blending (direct list) > prompt (converted to single-item list) > default
        """
        if not config_data:
            return None
            
        # Check for explicit prompt_blending first (highest priority)
        if 'prompt_blending' in config_data:
            prompt_blending = config_data['prompt_blending']
            
            # Handle nested structure: prompt_blending.prompt_list
            if isinstance(prompt_blending, dict) and 'prompt_list' in prompt_blending:
                prompt_list = prompt_blending['prompt_list']
                if isinstance(prompt_list, list) and len(prompt_list) > 0:
                    normalized = []
                    for item in prompt_list:
                        if isinstance(item, list) and len(item) == 2:
                            normalized.append([str(item[0]), float(item[1])])
                        elif isinstance(item, tuple) and len(item) == 2:
                            normalized.append([str(item[0]), float(item[1])])
                    if normalized:
                        return normalized
                        
            # Handle direct list format: prompt_blending: [["text", weight], ...]
            elif isinstance(prompt_blending, list) and len(prompt_blending) > 0:
                normalized = []
                for item in prompt_blending:
                    if isinstance(item, list) and len(item) == 2:
                        normalized.append([str(item[0]), float(item[1])])
                    elif isinstance(item, tuple) and len(item) == 2:
                        normalized.append([str(item[0]), float(item[1])])
                if normalized:
                    return normalized
        
        # Fall back to single prompt, convert to list format
        if 'prompt' in config_data:
            prompt = config_data['prompt']
            if isinstance(prompt, str) and prompt.strip():
                return [[prompt, 1.0]]  # Convert single prompt to list with weight 1.0
            elif isinstance(prompt, list) and len(prompt) > 0:
                # Handle case where prompt is already a list (but not in prompt_blending key)
                normalized = []
                for item in prompt:
                    if isinstance(item, list) and len(item) == 2:
                        normalized.append([str(item[0]), float(item[1])])
                    elif isinstance(item, tuple) and len(item) == 2:
                        normalized.append([str(item[0]), float(item[1])])
                    elif isinstance(item, str):
                        normalized.append([item, 1.0])
                if normalized:
                    return normalized
        
        return None

    def _normalize_seed_config(self, config_data):
        """
        Normalize seed configuration to always return a list format.
        Priority: seed_blending.seed_list > seed_blending (direct list) > seed (converted to single-item list) > default
        """
        if not config_data:
            return None
            
        # Check for explicit seed_blending first (highest priority)
        if 'seed_blending' in config_data:
            seed_blending = config_data['seed_blending']
            
            # Handle nested structure: seed_blending.seed_list
            if isinstance(seed_blending, dict) and 'seed_list' in seed_blending:
                seed_list = seed_blending['seed_list']
                if isinstance(seed_list, list) and len(seed_list) > 0:
                    normalized = []
                    for item in seed_list:
                        if isinstance(item, list) and len(item) == 2:
                            normalized.append([int(item[0]), float(item[1])])
                        elif isinstance(item, tuple) and len(item) == 2:
                            normalized.append([int(item[0]), float(item[1])])
                    if normalized:
                        return normalized
                        
            # Handle direct list format: seed_blending: [[seed, weight], ...]
            elif isinstance(seed_blending, list) and len(seed_blending) > 0:
                normalized = []
                for item in seed_blending:
                    if isinstance(item, list) and len(item) == 2:
                        normalized.append([int(item[0]), float(item[1])])
                    elif isinstance(item, tuple) and len(item) == 2:
                        normalized.append([int(item[0]), float(item[1])])
                if normalized:
                    return normalized
        
        # Fall back to single seed, convert to list format
        if 'seed' in config_data:
            seed = config_data['seed']
            if isinstance(seed, int):
                return [[seed, 1.0]]  # Convert single seed to list with weight 1.0
            elif isinstance(seed, list) and len(seed) > 0:
                # Handle case where seed is already a list (but not in seed_blending key)
                normalized = []
                for item in seed:
                    if isinstance(item, list) and len(item) == 2:
                        normalized.append([int(item[0]), float(item[1])])
                    elif isinstance(item, tuple) and len(item) == 2:
                        normalized.append([int(item[0]), float(item[1])])
                    elif isinstance(item, int):
                        normalized.append([item, 1.0])
                if normalized:
                    return normalized
        
        return None

    def _create_default_pipeline(self):
        """Create the default pipeline (standard mode)"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16
        pipeline = Pipeline(self.args, device, torch_dtype, width=self.new_width, height=self.new_height)
        
        # Initialize with default prompt blending (single prompt with weight 1.0)
        default_prompt = "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
        pipeline.stream.update_prompt([(default_prompt, 1.0)], prompt_interpolation_method="slerp")
        
        return pipeline

    def _create_pipeline_with_config(self, controlnet_config_path=None):
        """Create a new pipeline with optional ControlNet configuration"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16
        
        # Use runtime config if available (includes YAML + runtime additions), otherwise fallback to uploaded config
        if controlnet_config_path:
            new_args = self.args._replace(controlnet_config=controlnet_config_path)
        elif self.runtime_controlnet_config:
            # Use runtime config (includes YAML + runtime additions/removals)
            temp_config_path = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            yaml.dump(self.runtime_controlnet_config, temp_config_path, default_flow_style=False)
            temp_config_path.close()
            
            # Merge config values into args, respecting config overrides
            config_acceleration = self.runtime_controlnet_config.get('acceleration', self.args.acceleration)
            new_args = self.args._replace(
                controlnet_config=temp_config_path.name,
                acceleration=config_acceleration
            )
        elif self.uploaded_controlnet_config:
            # Fallback to original YAML config if no runtime modifications exist
            temp_config_path = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            yaml.dump(self.uploaded_controlnet_config, temp_config_path, default_flow_style=False)
            temp_config_path.close()
            
            # Merge YAML config values into args, respecting config overrides
            config_acceleration = self.uploaded_controlnet_config.get('acceleration', self.args.acceleration)
            new_args = self.args._replace(
                controlnet_config=temp_config_path.name,
                acceleration=config_acceleration
            )
        else:
            new_args = self.args
        
        new_pipeline = Pipeline(new_args, device, torch_dtype, width=self.new_width, height=self.new_height)
        
        # Initialize prompt blending from config (use runtime config if available)
        config_for_prompts = self.runtime_controlnet_config if self.runtime_controlnet_config else self.uploaded_controlnet_config
        normalized_prompt_config = self._normalize_prompt_config(config_for_prompts)
        if normalized_prompt_config:
            prompt_tuples = [(self._inject_trigger_words(item[0]), item[1]) for item in normalized_prompt_config]
            if self.lora_trigger_words:
                logger.info(f"_create_pipeline_with_config: injected trigger words into {len(prompt_tuples)} prompt(s)")
            new_pipeline.stream.update_prompt(prompt_tuples, prompt_interpolation_method="slerp")
        else:
            # Fallback to default single prompt
            default_prompt = "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
            new_pipeline.stream.update_prompt([(default_prompt, 1.0)], prompt_interpolation_method="slerp")
        
        # Apply style image (uploaded or default) if pipeline has IPAdapter
        has_ipadapter = getattr(new_pipeline, 'has_ipadapter', False)
        print(f"_create_pipeline_with_config: Pipeline has_ipadapter: {has_ipadapter}")
        
        if has_ipadapter:
            style_image = None
            style_source = ""
            
            if self.uploaded_style_image:
                style_image = self.uploaded_style_image
                style_source = "uploaded"
                print("_create_pipeline_with_config: Using uploaded style image")
            else:
                # Try to load default style image
                print("_create_pipeline_with_config: No uploaded style image, trying to load default")
                style_image = self._load_default_style_image()
                if style_image:
                    style_source = "default"
                    print("_create_pipeline_with_config: Default style image loaded successfully")
                else:
                    print("_create_pipeline_with_config: Failed to load default style image")
            
            if style_image:
                print(f"_create_pipeline_with_config: Applying {style_source} style image to new pipeline")
                success = new_pipeline.update_ipadapter_style_image(style_image)
                if success:
                    print(f"_create_pipeline_with_config: {style_source.capitalize()} style image applied successfully")
                    
                    # Force prompt re-encoding to apply style image embeddings
                    try:
                        state = new_pipeline.stream.get_stream_state()
                        current_prompts = state.get('prompt_list', [])
                        if current_prompts:
                            print("_create_pipeline_with_config: Forcing prompt re-encoding to apply style image")
                            new_pipeline.stream.update_prompt(current_prompts, prompt_interpolation_method="slerp")
                            print("_create_pipeline_with_config: Prompt re-encoding completed")
                    except Exception as e:
                        print(f"_create_pipeline_with_config: Failed to force prompt re-encoding: {e}")
                else:
                    print(f"_create_pipeline_with_config: Failed to apply {style_source} style image")
            else:
                print("_create_pipeline_with_config: No style image available (neither uploaded nor default)")

            # Re-apply style image B if one was uploaded
            if self.uploaded_style_image_b:
                success_b = new_pipeline.update_ipadapter_style_image_b(self.uploaded_style_image_b)
                print(f"_create_pipeline_with_config: Re-applied style image B: {success_b}")
        else:
            print("_create_pipeline_with_config: Pipeline does not have IPAdapter enabled")

        # Clean up temp file if created
        if self.uploaded_controlnet_config and not controlnet_config_path:
            try:
                os.unlink(new_args.controlnet_config)
            except:
                pass
        
        return new_pipeline

    def _get_controlnet_info(self):
        """Get ControlNet information from uploaded config or active pipeline"""
        controlnet_info = {
            "enabled": False,
            "config_loaded": False,
            "controlnets": []
        }
        
        # Check runtime config first (includes YAML + runtime additions/removals)
        if self.runtime_controlnet_config:
            controlnet_info["enabled"] = True
            controlnet_info["config_loaded"] = True
            if 'controlnets' in self.runtime_controlnet_config:
                for i, cn_config in enumerate(self.runtime_controlnet_config['controlnets']):
                    controlnet_info["controlnets"].append({
                        "index": i,
                        "name": cn_config['model_id'].split('/')[-1],
                        "preprocessor": cn_config['preprocessor'],
                        "strength": cn_config['conditioning_scale']
                    })
        # Fall back to uploaded YAML config if no runtime config exists
        elif self.uploaded_controlnet_config:
            controlnet_info["enabled"] = True
            controlnet_info["config_loaded"] = True
            if 'controlnets' in self.uploaded_controlnet_config:
                for i, cn_config in enumerate(self.uploaded_controlnet_config['controlnets']):
                    controlnet_info["controlnets"].append({
                        "index": i,
                        "name": cn_config['model_id'].split('/')[-1],
                        "preprocessor": cn_config['preprocessor'],
                        "strength": cn_config['conditioning_scale']
                    })
        # Otherwise check active pipeline
        elif self.pipeline and self.pipeline.use_config and self.pipeline.config and 'controlnets' in self.pipeline.config:
            controlnet_info["enabled"] = True
            controlnet_info["config_loaded"] = True
            if 'controlnets' in self.pipeline.config:
                for i, cn_config in enumerate(self.pipeline.config['controlnets']):
                    controlnet_info["controlnets"].append({
                        "index": i,
                        "name": cn_config['model_id'].split('/')[-1],
                        "preprocessor": cn_config['preprocessor'],
                        "strength": cn_config['conditioning_scale']
                    })
        
        return controlnet_info

    def _load_default_style_image(self):
        """Load the default style image for IPAdapter"""
        try:
            import os
            from PIL import Image
            
            default_image_path = os.path.join(os.path.dirname(__file__), "..", "..", "images", "inputs", "input.png")
            
            if os.path.exists(default_image_path):
                print(f"_load_default_style_image: Loading default style image (input.png) from {default_image_path}")
                return Image.open(default_image_path).convert("RGB")
            else:
                print(f"_load_default_style_image: Default style image not found at {default_image_path}")
                return None
                
        except Exception as e:
            print(f"_load_default_style_image: Failed to load default style image: {e}")
            return None

    def _get_ipadapter_info(self):
        """Get IPAdapter information from uploaded config or active pipeline"""
        ipadapter_info = {
            "enabled": False,
            "config_loaded": False,
            "scale": 1.0,
            "model_path": None,
            "style_image_set": False,
            "style_image_path": None
        }
        
        # Check uploaded config first
        if self.uploaded_controlnet_config:
            if 'ipadapters' in self.uploaded_controlnet_config and len(self.uploaded_controlnet_config['ipadapters']) > 0:
                ipadapter_info["enabled"] = True
                ipadapter_info["config_loaded"] = True
                
                # Get info from first IPAdapter config
                first_ipadapter = self.uploaded_controlnet_config['ipadapters'][0]
                ipadapter_info["scale"] = first_ipadapter.get('scale', DEFAULT_SETTINGS.get('ipadapter_scale', 1.0))
                ipadapter_info["model_path"] = first_ipadapter.get('ipadapter_model_path')
                
                # Check for style image - prioritize uploaded style image over config style image over default
                if self.uploaded_style_image:
                    ipadapter_info["style_image_set"] = True
                    ipadapter_info["style_image_path"] = "/api/ipadapter/uploaded-style-image"  # URL to fetch uploaded image
                elif 'style_image' in first_ipadapter:
                    ipadapter_info["style_image_set"] = True
                    ipadapter_info["style_image_path"] = first_ipadapter['style_image']
                else:
                    # Check if default image exists
                    import os
                    default_image_path = os.path.join(os.path.dirname(__file__), "..", "..", "images", "inputs", "input.png")
                    if os.path.exists(default_image_path):
                        ipadapter_info["style_image_set"] = True
                        ipadapter_info["style_image_path"] = "/api/default-image"
                    
        # Otherwise check active pipeline
        elif self.pipeline and self.pipeline.use_config and self.pipeline.config and 'ipadapters' in self.pipeline.config:
            if len(self.pipeline.config['ipadapters']) > 0:
                ipadapter_info["enabled"] = True
                ipadapter_info["config_loaded"] = True
                
                # Get info from first IPAdapter config
                first_ipadapter = self.pipeline.config['ipadapters'][0]
                ipadapter_info["scale"] = first_ipadapter.get('scale', DEFAULT_SETTINGS.get('ipadapter_scale', 1.0))
                ipadapter_info["model_path"] = first_ipadapter.get('ipadapter_model_path')
                
                # Check for style image - prioritize uploaded style image over config style image over default
                if self.uploaded_style_image:
                    ipadapter_info["style_image_set"] = True
                    ipadapter_info["style_image_path"] = "/api/ipadapter/uploaded-style-image"  # URL to fetch uploaded image
                elif 'style_image' in first_ipadapter:
                    ipadapter_info["style_image_set"] = True
                    ipadapter_info["style_image_path"] = first_ipadapter['style_image']
                else:
                    # Check if default image exists
                    import os
                    default_image_path = os.path.join(os.path.dirname(__file__), "..", "..", "images", "inputs", "input.png")
                    if os.path.exists(default_image_path):
                        ipadapter_info["style_image_set"] = True
                        ipadapter_info["style_image_path"] = "/api/default-image"
                    
            # Try to get current scale from active pipeline if available
            try:
                if hasattr(self.pipeline, 'get_ipadapter_info'):
                    pipeline_info = self.pipeline.get_ipadapter_info()
                    if pipeline_info.get("enabled"):
                        ipadapter_info["scale"] = pipeline_info.get("scale", ipadapter_info["scale"])
            except:
                pass
        
        return ipadapter_info

    def _inject_trigger_words(self, prompt_text: str) -> str:
        """Prepend active LoRA trigger words to a prompt, skipping if already present."""
        if not self.lora_trigger_words:
            return prompt_text
        if self.lora_trigger_words.lower() in prompt_text.lower():
            return prompt_text
        return f"{self.lora_trigger_words}, {prompt_text}"

    def _calculate_aspect_ratio(self, width: int, height: int) -> str:
        """Calculate and return aspect ratio as a string"""
        import math
        
        # Find GCD to simplify the ratio
        gcd = math.gcd(width, height)
        simplified_width = width // gcd
        simplified_height = height // gcd
        
        return f"{simplified_width}:{simplified_height}"

    def _cleanup_pipeline(self, pipeline):
        """Properly cleanup a pipeline and free VRAM using StreamDiffusion's built-in cleanup"""
        if pipeline is None:
            return
            
        try:
            logger.info("Starting pipeline cleanup...")
            
            # Use StreamDiffusion's built-in cleanup method which properly handles:
            # - TensorRT engine cleanup
            # - ControlNet engine cleanup  
            # - Multiple garbage collection cycles
            # - CUDA cache clearing
            # - Memory tracking
            if hasattr(pipeline, 'stream') and pipeline.stream and hasattr(pipeline.stream, 'cleanup_gpu_memory'):
                pipeline.stream.cleanup_gpu_memory()
                logger.info("Pipeline cleanup completed using StreamDiffusion cleanup")
            else:
                # Fallback cleanup if the method doesn't exist
                logger.warning("StreamDiffusion cleanup method not found, using fallback cleanup")
                if hasattr(pipeline, 'stream') and pipeline.stream:
                    del pipeline.stream
                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            logger.error(f"Error during pipeline cleanup: {e}")
            # Still try to clear CUDA cache even if cleanup fails
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _update_resolution(self, width: int, height: int) -> None:
        """Create a new pipeline with the specified resolution and replace the old one."""
        logger.info(f"Creating new pipeline with resolution {width}x{height}")
        
        # Store current pipeline state before cleanup
        current_prompt = getattr(self.pipeline, 'prompt', '') if self.pipeline else ''
        current_negative_prompt = getattr(self.pipeline, 'negative_prompt', '') if self.pipeline else ''
        current_guidance_scale = getattr(self.pipeline, 'guidance_scale', 1.2) if self.pipeline else 1.2
        current_num_inference_steps = getattr(self.pipeline, 'num_inference_steps', 50) if self.pipeline else 50
        
        # Store reference to old pipeline for cleanup
        old_pipeline = self.pipeline
        
        # Clear current pipeline reference before cleanup to prevent any access during cleanup
        self.pipeline = None
        
        # Cleanup old pipeline and free VRAM
        if old_pipeline:
            self._cleanup_pipeline(old_pipeline)
            old_pipeline = None
        
        # Update current resolution 
        self.new_width = width
        self.new_height = height
        
        # Create new pipeline with new resolution
        try:
            if self.uploaded_controlnet_config:
                new_pipeline = self._create_pipeline_with_config()
            else:
                new_pipeline = self._create_default_pipeline()
            
            # Apply style image (uploaded or default) if pipeline has IPAdapter
            has_ipadapter = getattr(new_pipeline, 'has_ipadapter', False)
            print(f"_update_resolution: Pipeline has_ipadapter: {has_ipadapter}")
            
            if has_ipadapter:
                style_image = None
                style_source = ""
                
                if self.uploaded_style_image:
                    style_image = self.uploaded_style_image
                    style_source = "uploaded"
                    print("_update_resolution: Using uploaded style image")
                else:
                    # Try to load default style image
                    print("_update_resolution: No uploaded style image, trying to load default")
                    style_image = self._load_default_style_image()
                    if style_image:
                        style_source = "default"
                        print("_update_resolution: Default style image loaded successfully")
                    else:
                        print("_update_resolution: Failed to load default style image")
                
                if style_image:
                    print(f"_update_resolution: Applying {style_source} style image to new pipeline")
                    success = new_pipeline.update_ipadapter_style_image(style_image)
                    if success:
                        print(f"_update_resolution: {style_source.capitalize()} style image applied successfully")
                        
                        # Force prompt re-encoding to apply style image embeddings
                        try:
                            state = new_pipeline.stream.get_stream_state()
                            current_prompts = state.get('prompt_list', [])
                            if current_prompts:
                                print("_update_resolution: Forcing prompt re-encoding to apply style image")
                                new_pipeline.stream.update_prompt(current_prompts, prompt_interpolation_method="slerp")
                                print("_update_resolution: Prompt re-encoding completed")
                        except Exception as e:
                            print(f"_update_resolution: Failed to force prompt re-encoding: {e}")
                    else:
                        print(f"_update_resolution: Failed to apply {style_source} style image")
                else:
                    print("_update_resolution: No style image available (neither uploaded nor default)")

                # Re-apply style image B if one was uploaded
                if self.uploaded_style_image_b:
                    success_b = new_pipeline.update_ipadapter_style_image_b(self.uploaded_style_image_b)
                    print(f"_update_resolution: Re-applied style image B: {success_b}")
            else:
                print("_update_resolution: Pipeline does not have IPAdapter enabled")

            # Set the new pipeline
            self.pipeline = new_pipeline
            
            # Restore pipeline state
            if current_prompt:
                self.pipeline.stream.prepare(
                    prompt=current_prompt,
                    negative_prompt=current_negative_prompt,
                    guidance_scale=current_guidance_scale,
                    num_inference_steps=current_num_inference_steps
                )
                # Also update the pipeline's stored values
                self.pipeline.prompt = current_prompt
                self.pipeline.negative_prompt = current_negative_prompt
                self.pipeline.guidance_scale = current_guidance_scale
                self.pipeline.num_inference_steps = current_num_inference_steps
                self.pipeline.last_prompt = current_prompt
            
            logger.info(f"Pipeline updated successfully to {width}x{height}")
            
        except Exception as e:
            logger.error(f"Failed to create new pipeline: {e}")
            # Make sure we don't leave the system in a broken state
            self.pipeline = None
            raise

app = App(config).app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        ssl_certfile=config.ssl_certfile,
        ssl_keyfile=config.ssl_keyfile,
    )
