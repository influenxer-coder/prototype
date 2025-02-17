import base64
import io
import cv2
from PIL import Image
import numpy as np

import json
import re

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


def extract_json(response_text: str) -> dict:
    try:
        # Remove any text before and after the JSON
        # Look for the first { and last }
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON structure found in response")

        json_str = response_text[start_idx:end_idx]

        # Clean the string if needed
        json_str = json_str.strip()

        # Parse JSON
        data = json.loads(json_str)

        return data

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        raise
    except ValueError as e:
        print(f"Validation error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


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