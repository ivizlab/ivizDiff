"""
Quick diagnostic: does the user LoRA actually change UNet weights when fused?
Run with: python utils/test_lora.py
"""
import sys
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

LORA_PATH = "C:/AI/models/Lora15/sdsdsd1_sd15.safetensors"
MODEL_ID = "C:/AI/models/Stable-diffusion/PerfectPhotonV2.1"

def _lora_has_text_encoder_keys(lora_path):
    try:
        from safetensors import safe_open
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            te_keys = [k for k in keys if "text_encoder" in k or k.startswith("lora_te")]
            unet_keys = [k for k in keys if k.startswith("lora_unet")]
            logger.info(f"LoRA total keys: {len(keys)}")
            logger.info(f"Sample keys: {keys[:5]}")
            logger.info(f"UNet keys: {len(unet_keys)}, TE keys: {len(te_keys)}")
            return len(te_keys) > 0
    except Exception as e:
        logger.error(f"Could not inspect LoRA: {e}")
        return True

def main():
    logger.info(f"Loading pipeline from {MODEL_ID}...")
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    pipe = pipe.to("cpu")

    # Use cross-attention probe (most LoRAs target attn layers, not conv_in)
    sd = pipe.unet.state_dict()
    probe_key = next(
        (k for k in sd if "attn2.to_k.weight" in k or "attn1.to_k.weight" in k),
        next(iter(sd))
    )
    before = sd[probe_key].clone()
    logger.info(f"Probe weight key: {probe_key}")
    logger.info(f"Before fuse — first 3 values: {before.flatten()[:3]}")

    has_te = _lora_has_text_encoder_keys(LORA_PATH)
    components = ["unet"] + (["text_encoder"] if has_te else [])
    logger.info(f"has_te={has_te}, components={components}")

    logger.info(f"Loading LoRA from {LORA_PATH}...")
    try:
        pipe.load_lora_weights(LORA_PATH)
        logger.info("load_lora_weights: OK")
    except Exception as e:
        logger.error(f"load_lora_weights FAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    logger.info(f"Calling fuse_lora(lora_scale=1.0, components={components})...")
    try:
        pipe.fuse_lora(lora_scale=1.0, components=components)
        logger.info("fuse_lora: OK")
    except Exception as e:
        logger.error(f"fuse_lora FAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    after = pipe.unet.state_dict()[probe_key].clone()
    logger.info(f"After fuse  — first 3 values: {after.flatten()[:3]}")
    changed = not torch.allclose(before.float(), after.float(), atol=1e-6)
    logger.info(f"UNet weights changed: {changed}")
    if not changed:
        logger.error("WEIGHTS DID NOT CHANGE — LoRA fuse had no effect!")
        logger.error("Possible causes:")
        logger.error("  1. diffusers fuse_lora bug (check diffusers version)")
        logger.error("  2. LoRA keys don't overlap with probed layer")
        logger.error("  3. PEFT adapter was not installed correctly")
    else:
        logger.info("SUCCESS — LoRA UNet fuse is working correctly.")

    try:
        pipe.unload_lora_weights()
    except Exception:
        pass

if __name__ == "__main__":
    main()
