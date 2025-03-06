import base64
import io
from typing import List

import cv2
import numpy as np
from PIL import Image

from app.config.settings import Config


def frame_to_base64(frame: np.ndarray) -> dict:
    """Convert frame to base64 for API transmission."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=95)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": image_base64
        }
    }


def get_video_duration_cv2(video_path: str) -> float:
    """
    Get video duration using OpenCV.
    Returns duration in seconds.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = frame_count / fps
    cap.release()

    return duration


def extract_hook_frame(video_path: str, frame_time: int = 1) -> np.ndarray | None:
    """
    Extract a specific frame from a video file

    Args:
        frame_time (int): Time to extract the frame (default: 1)
        video_path (str): Path to the video file

    Returns:
        numpy.ndarray: The extracted frame, or None if extraction failed
    """

    frame = None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = fps * frame_time

    current_frame = 0
    while current_frame < frame_number:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Video has fewer than {frame_number} frames")
            cap.release()
            return None
        current_frame += 1

    cap.release()
    return frame


def extract_keyframes(video_path: str, max_duration_seconds: float = 5.0) -> List[tuple]:
    """
    Extract keyframes from video based on scene changes up to a specified duration.

    Args:
        video_path: Path to the video file
        max_duration_seconds: Maximum duration in seconds to extract keyframes from (None for entire video)

    Returns:
        List of tuples containing (frame_number, timestamp, frame)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    min_frame_interval = int(fps * Config.MIN_INTERVAL_SECONDS)

    keyframes = []
    prev_frame = None
    frames_since_last_keyframe = 0
    frame_number = 0
    frames_extracted = {}

    # Calculate the maximum frame number based on duration if specified
    max_frame = None
    if max_duration_seconds is not None:
        max_frame = int(max_duration_seconds * fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if we have reached the maximum duration
        if max_frame is not None and frame_number >= max_frame:
            break

        if prev_frame is None:
            keyframes.append((frame_number, frame_number / fps, frame.copy()))
            frames_since_last_keyframe = 0
            frames_extracted[frame_number] = True
        elif frames_since_last_keyframe >= min_frame_interval:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(curr_gray, prev_gray)
            mean_diff = np.mean(frame_diff)

            if mean_diff > Config.MIN_SCENE_CHANGE_THRESHOLD:
                keyframes.append((frame_number, frame_number / fps, frame.copy()))
                frames_since_last_keyframe = 0
                frames_extracted[frame_number] = True

        prev_frame = frame.copy()
        frame_number += 1
        frames_since_last_keyframe += 1

    if frame_number not in frames_extracted and prev_frame is not None:
        keyframes.append((frame_number, frame_number / fps, prev_frame.copy()))

    cap.release()
    return keyframes
