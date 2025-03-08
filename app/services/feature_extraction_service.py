import os
from typing import Optional, List

import cv2

from app.models.video import KeyframeContext
from app.services.audio.audio_service import AudioService
from app.services.visual.visual_service import VideoService
from app.services.client.llm_agent_service import LlmAgentService
from app.utils.transcript import get_audio_hook


class FeatureExtractionService:
    def __init__(self):
        self.llm = LlmAgentService()
        self.audio_service = AudioService()
        self.visual_servie = VideoService()

    def get_video_duration(self, video_path: str) -> float:
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

    def get_keyframes(self, video_path: str, max_duration_seconds: Optional[float] = None) -> List[tuple]:
        return self.visual_servie.extract_keyframes(video_path, max_duration_seconds)

    def get_visual_features(self, video_path: str):
        video_duration = self.get_video_duration(video_path)
        keyframes = self.get_keyframes(video_path, min(video_duration, 5.0))

        keyframe_contexts = [
            KeyframeContext(
                frame_number=i + 1,
                timestamp=timestamp,
                image=frame,
                audio_transcript=None,
                window_start=0 if i == 0 else keyframes[i - 1][1],
                window_end=timestamp
            )
            for i, (frame_num, timestamp, frame) in enumerate(keyframes)
        ]

        # call summary generator
        print("Calling AGENT to generate visual features...")
        visual_features = self.llm.generate_visual_features(keyframe_contexts)
        return visual_features

    def get_style_features(self, video_path: str):
        keyframes = self.get_keyframes(video_path)
        print(f"Extracted {len(keyframes)} keyframes")

        analysis = self.llm.generate_style_features(keyframes)

        creator_visible = analysis.get("creator_visible", None)
        product_visible = analysis.get("product_visible", None)

        os.remove(video_path)

        print("-" * 80)
        print(f"Keyframe Analysis:")
        print(f"\tCreator Visibility: {creator_visible}")
        print(f"\tProduct Visibility: {product_visible}")
        print("-" * 80)

        return {
            "creator_visible": creator_visible,
            "product_visible": product_visible
        }

    def transcribe(self, audio_path: str, start_time: float | None = None, end_time: float | None = None) -> str:
        return self.audio_service.transcribe(audio_path, start_time, end_time)

    def get_visual_hook(self, video_file_path: str, full_script: str):
        frame = self.visual_servie.extract_hook_frame(video_file_path, frame_time=1)
        print("Calling AGENT to generate screen hook...")
        screen_hook = self.llm.generate_screen_hook(frame)

        audio_hook = get_audio_hook(full_script)
        print("Calling AGENT to analyze hook...")
        shooting_style = self.llm.generate_hook_analysis(frame, full_script)

        return {
            "screen_hook": screen_hook,
            "audio_hook": audio_hook,
            "shooting_style": shooting_style
        }

