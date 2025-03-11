import base64
import io
from typing import List, Optional

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