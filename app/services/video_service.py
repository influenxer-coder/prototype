import os
import tempfile

from app.models.video import KeyframeAudioContext
from app.services.audio_service import AudioService
from app.services.llm_agent_service import LlmAgentService
from app.utils.audio import extract_audio
from app.utils.video import extract_keyframes


class VideoService:
    def __init__(self):
        self.audio_service = AudioService()
        self.llm_agent_service = LlmAgentService()

    def process_video(self, video_path: str, caption: str):
        """Process video and generate analysis."""
        print("Extracting keyframes...")
        keyframes = extract_keyframes(video_path)
        print(f"Found {len(keyframes)} keyframes")

        audio_dir = tempfile.gettempdir()
        filename = os.path.basename(video_path)
        audio_path = f"{audio_dir}/{os.path.splitext(filename)[0]}.wav"

        if not extract_audio(video_path, audio_path):
            print(f"Error in extracting audio to {audio_path}")
            raise ValueError(f"Unable to extract audio from {video_path}")

        complete_transcript = self.audio_service.transcribe(audio_path)

        print("Processing audio for each keyframe...")
        keyframe_contexts = []

        for i, (frame_num, timestamp, frame) in enumerate(keyframes):
            print(f"Processing keyframe {i + 1}/{len(keyframes)}")
            start_time = 0 if i == 0 else keyframes[i - 1][1]

            audio_transcript = self.audio_service.transcribe(
                audio_path,
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

        # clean up
        os.remove(audio_path)

        # call summary generator
        print("Calling AGENT to generate summary...")
        summary = self.llm_agent_service.generate_summary(keyframe_contexts, caption)

        # call screenplay generator
        print("Calling AGENT to generate screenplay...")
        screenplay = self.llm_agent_service.generate_screenplay(summary, complete_transcript)

        return summary, screenplay
