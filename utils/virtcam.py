"""
virtcam.py — Feed a local video file into OBS Virtual Camera.

Usage:
    python utils/virtcam.py <video_path> [--width W] [--height H] [--speed N]

Reads the video in a loop, resizes each frame to the camera's native
resolution (default 512×512 to match the pipeline), and pushes it into
the OBS Virtual Camera DirectShow device via pyvirtualcam.

--speed N: playback slowdown factor (1 = normal, 2 = half speed, 5 = 5× slower).
           Each video frame is held for N camera frames.

Requirements:
    - OBS Virtual Camera driver installed (comes with OBS Studio)
    - pyvirtualcam: already in the SDiff conda env
"""

import sys
import argparse
import time
import cv2
import numpy as np

try:
    import pyvirtualcam
except ImportError:
    print("ERROR: pyvirtualcam not installed. Run: pip install pyvirtualcam", flush=True)
    sys.exit(1)


def run(video_path: str, width: int = 512, height: int = 512, fps: float = 30.0, speed_factor: int = 1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}", flush=True)
        sys.exit(1)

    # Use the video's native FPS if available
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps and native_fps > 0:
        fps = native_fps

    repeat = max(1, int(speed_factor))
    print(f"virtcam: starting  {video_path}  {width}x{height} @ {fps:.1f}fps  speed={repeat}x slower", flush=True)

    try:
        with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
            print(f"virtcam: sending to {cam.device}", flush=True)
            while True:
                ok, frame = cap.read()
                if not ok:
                    # Loop: rewind to start
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok:
                        print("virtcam: failed to rewind video", flush=True)
                        break

                # Resize to target resolution (centre-crop to maintain aspect ratio)
                fh, fw = frame.shape[:2]
                scale = max(width / fw, height / fh)
                new_w, new_h = int(fw * scale), int(fh * scale)
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                x0 = (new_w - width) // 2
                y0 = (new_h - height) // 2
                cropped = resized[y0:y0 + height, x0:x0 + width]

                # Hold each video frame for `repeat` camera frames (slowdown)
                for _ in range(repeat):
                    cam.send(cropped)
                    cam.sleep_until_next_frame()

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print("virtcam: stopped", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feed a video file into OBS Virtual Camera")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--speed", type=int, default=1,
                        help="Slowdown factor: 1=normal, 2=half speed, 5=5× slower")
    # Legacy positional width/height for backwards compat with old Popen calls
    parser.add_argument("pos_width", nargs="?", type=int, help=argparse.SUPPRESS)
    parser.add_argument("pos_height", nargs="?", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    w = args.pos_width if args.pos_width is not None else args.width
    h = args.pos_height if args.pos_height is not None else args.height

    run(args.video_path, w, h, speed_factor=args.speed)
