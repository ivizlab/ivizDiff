"""
Video Input Manager for StreamDiffusion
Handles MP4 video file input with looping capability
"""

import cv2
import threading
import time
import logging
from typing import Optional, Callable
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class VideoInputManager:
    def __init__(self):
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.video_path: Optional[str] = None
        self.is_playing = False
        self.is_looping = True
        self.fps = 30.0
        self.frame_callback: Optional[Callable] = None
        self.playback_thread: Optional[threading.Thread] = None
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        
    def load_video(self, video_path: str) -> bool:
        """
        Load an MP4 video file
        
        Args:
            video_path: Path to the MP4 file
            
        Returns:
            bool: True if video loaded successfully, False otherwise
        """
        try:
            if not Path(video_path).exists():
                logger.error(f"Video file not found: {video_path}")
                return False
                
            # Stop any existing playback
            self.stop()
            
            # Initialize video capture
            self.video_capture = cv2.VideoCapture(video_path)
            
            if not self.video_capture.isOpened():
                logger.error(f"Failed to open video file: {video_path}")
                return False
                
            # Get video properties
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / self.fps
            
            self.video_path = video_path
            
            logger.info(f"Video loaded: {video_path}")
            logger.info(f"Properties: {frame_count} frames, {self.fps:.2f} FPS, {duration:.2f}s duration")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            return False
    
    def set_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """
        Set callback function to receive video frames
        
        Args:
            callback: Function that will be called with each frame (numpy array)
        """
        self.frame_callback = callback
    
    def start_playback(self, loop: bool = True) -> bool:
        """
        Start video playback
        
        Args:
            loop: Whether to loop the video when it reaches the end
            
        Returns:
            bool: True if playback started successfully, False otherwise
        """
        if not self.video_capture or not self.video_capture.isOpened():
            logger.error("No video loaded or video capture not opened")
            return False
            
        if self.is_playing:
            logger.warning("Video playback already started")
            return True
            
        self.is_looping = loop
        self.is_playing = True
        
        # Start playback thread
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()
        
        logger.info("Video playback started")
        return True
    
    def stop(self):
        """Stop video playback"""
        self.is_playing = False
        
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)
            
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
            
        self.current_frame = None
        logger.info("Video playback stopped")


    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get the current frame
        
        Returns:
            np.ndarray: Current frame as BGR image, or None if no frame available
        """
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def _playback_loop(self):
        """Internal playback loop running in separate thread"""
        frame_duration = 1.0 / self.fps
        
        while self.is_playing:
            start_time = time.time()
            
            ret, frame = self.video_capture.read()
            
            if not ret:
                if self.is_looping:
                    # Reset to beginning for looping
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.video_capture.read()
                    
                if not ret:
                    logger.error("Failed to read frame, stopping playback")
                    break
            
            if frame is not None:
                # Store current frame
                with self.frame_lock:
                    self.current_frame = frame
                
                # Call frame callback if set
                if self.frame_callback:
                    try:
                        self.frame_callback(frame)
                    except Exception as e:
                        logger.error(f"Error in frame callback: {e}")
            
            # Maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_duration - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.is_playing = False
        logger.info("Playback loop ended")
    
    def get_video_info(self) -> dict:
        """
        Get information about the currently loaded video
        
        Returns:
            dict: Video information including path, fps, frame count, etc.
        """
        if not self.video_capture or not self.video_capture.isOpened():
            return {}
            
        frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / self.fps if self.fps > 0 else 0
        
        return {
            "path": self.video_path,
            "fps": self.fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration,
            "is_playing": self.is_playing,
            "is_looping": self.is_looping
        }

    def frame_to_bytes(self, frame: np.ndarray, format: str = 'JPEG') -> bytes:
        """
        Convert frame to bytes for transmission
        
        Args:
            frame: OpenCV frame (BGR format)
            format: Image format ('JPEG' or 'PNG')
            
        Returns:
            bytes: Encoded image bytes
        """
        if format.upper() == 'JPEG':
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            result, encoded_img = cv2.imencode('.jpg', frame, encode_param)
        else:
            result, encoded_img = cv2.imencode('.png', frame)
            
        if result:
            return encoded_img.tobytes()
        else:
            raise ValueError("Failed to encode frame")
