from importlib import import_module
from types import ModuleType
from typing import Dict, Any
from pydantic import BaseModel as PydanticBaseModel, Field
from PIL import Image
import io
import torch
import numpy as np
import cv2
from torchvision.io import encode_jpeg, decode_jpeg


def get_pipeline_class(pipeline_name: str) -> ModuleType:
    try:
        module = import_module(f"pipelines.{pipeline_name}")
    except ModuleNotFoundError:
        raise ValueError(f"Pipeline {pipeline_name} module not found")

    pipeline_class = getattr(module, "Pipeline", None)

    if pipeline_class is None:
        raise ValueError(f"'Pipeline' class not found in module '{pipeline_name}'.")

    return pipeline_class


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    return image


def bytes_to_pt(image_bytes: bytes) -> torch.Tensor:
    """
    Convert JPEG/PNG bytes directly to PyTorch tensor using torchvision
    
    Args:
        image_bytes: Raw image bytes (JPEG/PNG format)
        
    Returns:
        torch.Tensor: Image tensor with shape (C, H, W), values in [0, 1], dtype float32
    """
    # Convert bytes to tensor for torchvision
    byte_tensor = torch.frombuffer(image_bytes, dtype=torch.uint8)
    
    # Decode JPEG/PNG directly to tensor (C, H, W) format, uint8 [0, 255]
    image_tensor = decode_jpeg(byte_tensor)
    
    # Convert to float32 and normalize to [0, 1]
    image_tensor = image_tensor.float() / 255.0
    
    return image_tensor


def pil_to_frame(image: Image.Image) -> bytes:
    frame_data = io.BytesIO()
    image.save(frame_data, format="JPEG")
    frame_data = frame_data.getvalue()
    return (
        b"--frame\r\n"
        + b"Content-Type: image/jpeg\r\n"
        + f"Content-Length: {len(frame_data)}\r\n\r\n".encode()
        + frame_data
        + b"\r\n"
    )


def pt_to_frame(tensor: torch.Tensor) -> bytes:
    """
    Convert PyTorch tensor directly to JPEG frame bytes using torchvision
    
    Args:
        tensor: PyTorch tensor with shape (C, H, W) or (1, C, H, W), values in [0, 1]
        
    Returns:
        bytes: JPEG frame data for streaming
    """
    # Handle batch dimension - take first image if batched
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Convert to uint8 format (0-255) and ensure correct shape (C, H, W)
    tensor_uint8 = (tensor * 255).clamp(0, 255).to(torch.uint8)
    
    # Encode directly to JPEG bytes using torchvision
    jpeg_bytes = encode_jpeg(tensor_uint8, quality=90)
    frame_data = jpeg_bytes.cpu().numpy().tobytes()
    
    return (
        b"--frame\r\n"
        + b"Content-Type: image/jpeg\r\n"
        + f"Content-Length: {len(frame_data)}\r\n\r\n".encode()
        + frame_data
        + b"\r\n"
    )


def is_firefox(user_agent: str) -> bool:
    return "Firefox" in user_agent


# NEW VIDEO FUNCTIONS - ADD THESE TO YOUR EXISTING util.py
def cv_frame_to_pt(cv_frame: np.ndarray) -> torch.Tensor:
    """
    Convert OpenCV frame (BGR) directly to PyTorch tensor
    
    Args:
        cv_frame: OpenCV frame in BGR format (H, W, C), dtype uint8
        
    Returns:
        torch.Tensor: Image tensor with shape (C, H, W), values in [0, 1], dtype float32
    """
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor and change from (H, W, C) to (C, H, W)
    tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float()
    
    # Normalize to [0, 1]
    tensor = tensor / 255.0
    
    return tensor


def cv_frame_to_pil(cv_frame: np.ndarray) -> Image.Image:
    """
    Convert OpenCV frame (BGR) to PIL Image
    
    Args:
        cv_frame: OpenCV frame in BGR format (H, W, C), dtype uint8
        
    Returns:
        PIL.Image: RGB image
    """
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    
    return pil_image


def resize_cv_frame(cv_frame: np.ndarray, target_width: int, target_height: int, 
                   maintain_aspect_ratio: bool = False) -> np.ndarray:
    """
    Resize OpenCV frame to target dimensions
    
    Args:
        cv_frame: OpenCV frame (H, W, C)
        target_width: Target width
        target_height: Target height
        maintain_aspect_ratio: Whether to maintain aspect ratio (will pad with black if needed)
        
    Returns:
        np.ndarray: Resized frame
    """
    if not maintain_aspect_ratio:
        # Simple resize
        return cv2.resize(cv_frame, (target_width, target_height))
    
    # Maintain aspect ratio with padding
    h, w = cv_frame.shape[:2]
    aspect_ratio = w / h
    target_aspect_ratio = target_width / target_height
    
    if aspect_ratio > target_aspect_ratio:
        # Image is wider than target
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Image is taller than target
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    
    # Resize to fit within target dimensions
    resized = cv2.resize(cv_frame, (new_width, new_height))
    
    # Create black background
    result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Center the resized image
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    result[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
    
    return result


def bytes_to_pt_enhanced(image_bytes: bytes, source_format: str = "jpeg") -> torch.Tensor:
    """
    Enhanced version that can handle different input formats
    
    Args:
        image_bytes: Raw image bytes
        source_format: "jpeg", "png", or "bgr_raw"
        
    Returns:
        torch.Tensor: Image tensor with shape (C, H, W), values in [0, 1], dtype float32
    """
    if source_format in ["jpeg", "png"]:
        # Use existing JPEG/PNG decoding
        return bytes_to_pt(image_bytes)
    elif source_format == "bgr_raw":
        # Handle raw BGR data from OpenCV
        # This would need additional width/height information
        # For now, fall back to JPEG decoding
        return bytes_to_pt(image_bytes)
    else:
        raise ValueError(f"Unsupported source format: {source_format}")
