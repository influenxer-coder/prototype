from typing import List

import cv2
import numpy as np

from app.config.settings import Config
from app.models.video import KeyframeAudioContext
from app.services.audio_analytics_service import AudioAnalyticsService
from app.services.llm_agent_service import LlmAgentService
from app.utils.video import get_video_duration_cv2


class VideoAnalyticsService:
    def __init__(self):
        self.audio_analytics_service = AudioAnalyticsService()
        self.llm_agent_service = LlmAgentService()

    def extract_keyframes(self, video_path: str) -> List[tuple]:
        """Extract keyframes from video based on scene changes."""
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

        while True:
            ret, frame = cap.read()
            if not ret:
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

        if frame_number not in frames_extracted:
            keyframes.append((frame_number, frame_number / fps, prev_frame.copy()))

        cap.release()
        return keyframes

    def process_video(self, video_path: str, caption: str):
        """Process video and generate analysis."""
        print("Extracting keyframes...")
        keyframes = self.extract_keyframes(video_path)
        print(f"Found {len(keyframes)} keyframes")

        video_duration = get_video_duration_cv2(video_path)

        complete_transcript = self.audio_analytics_service.get_transcript(
            video_path,
            start_time=0,
            end_time=video_duration
        )

        print("Processing audio for each keyframe...")
        keyframe_contexts = []

        for i, (frame_num, timestamp, frame) in enumerate(keyframes):
            print(f"Processing keyframe {i + 1}/{len(keyframes)}")
            start_time = 0 if i == 0 else keyframes[i - 1][1]

            audio_transcript = self.audio_analytics_service.get_transcript(
                video_path,
                start_time,
                timestamp
            )

            context = KeyframeAudioContext(
                frame_number=i + 1,
                timestamp=timestamp,
                image=frame,
                audio_transcript=audio_transcript,
                window_start=start_time,
                window_end=timestamp
            )
            keyframe_contexts.append(context)

        # call summary generator
        print("Calling AGENT to generate summary...")
        summary = self.llm_agent_service.generate_summary(keyframe_contexts, caption)

        # call screenplay generator
        print("Calling AGENT to generate screenplay...")
        screenplay = self.llm_agent_service.generate_screenplay(summary, complete_transcript)

        return summary, screenplay
