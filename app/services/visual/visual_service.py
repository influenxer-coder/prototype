from typing import Optional, List

import cv2
import numpy as np

from app import Config


class VideoService:
    def __init__(self):
        pass

    def _is_scene_change(self, frame, prev_frame, threshold=Config.MIN_SCENE_CHANGE_THRESHOLD) -> bool:
        """
        Helper function to detect scene changes between two frames.

        Args:
            frame: Current frame.
            prev_frame: Previous frame.
            threshold: Minimum mean difference to consider a scene change.

        Returns:
            True if a scene change is detected, False otherwise.
        """
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(curr_gray, prev_gray)
        return np.mean(frame_diff) > threshold

    def extract_keyframes(self, video_path: str, max_duration_seconds: Optional[float] = None) -> List[tuple]:
        """
        Extract keyframes from a video based on scene changes, up to a specified duration.

        Args:
            video_path: Path to the video file.
            max_duration_seconds: Maximum duration in seconds to extract keyframes from (None for the entire video).

        Returns:
            List of tuples containing (frame_number, timestamp, frame).
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        min_frame_interval = int(fps * Config.MIN_INTERVAL_SECONDS)
        max_frame = int(max_duration_seconds * fps) if max_duration_seconds else None

        # Initialize variables
        keyframes = []
        prev_frame = None
        frames_extracted = set()
        frame_number = 0
        frames_since_last_keyframe = 0

        while True:
            ret, frame = cap.read()
            if not ret or (max_frame and frame_number >= max_frame):
                break  # Stop if video ends or max duration is reached

            # Check if it's the first frame or a significant scene change
            if prev_frame is None or (
                    frames_since_last_keyframe >= min_frame_interval
                    and self._is_scene_change(frame, prev_frame)
            ):
                keyframes.append((frame_number, frame_number / fps, frame.copy()))
                frames_extracted.add(frame_number)
                frames_since_last_keyframe = 0  # Reset counter

            # Update tracking variables
            prev_frame = frame.copy()
            frame_number += 1
            frames_since_last_keyframe += 1

        # Ensure the last frame is included if it wasn't already
        if prev_frame is not None and frame_number not in frames_extracted:
            keyframes.append((frame_number, frame_number / fps, prev_frame.copy()))

        cap.release()
        return keyframes

    def extract_hook_frame(self, video_path: str, frame_time: int = 1) -> Optional[np.ndarray]:
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
