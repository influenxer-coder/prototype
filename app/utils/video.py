import base64
import io

import cv2
import numpy as np
from PIL import Image


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
