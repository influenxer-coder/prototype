import os
import random
import tempfile
import time

from pandas.core.frame import DataFrame

from app.models.video import KeyframeContext
from app.services.audio_service import AudioService
from app.services.llm_agent_service import LlmAgentService
from app.services.recommendation_service import RecommendationService
from app.services.s3_service import S3Service
from app.services.scraper_service import ScraperService
from app.utils.audio import extract_audio
from app.utils.dataframe import calculate_impact_scores
from app.utils.transcript import get_audio_hook
from app.utils.video import extract_hook_frame, extract_keyframes, get_video_duration_cv2


class IngestionService:
    def __init__(self):
        self.s3 = S3Service()
        self.scraper = ScraperService()
        self.audio = AudioService()
        self.llm = LlmAgentService()
        self.video = RecommendationService()
        self.video_bucket = "tapestry-tiktok-videos"

    def process(self, posts: DataFrame) -> DataFrame:
        # Download videos from TikTok
        video_linked_posts = self.download_videos(posts)

        # Calculate impact scores
        scored_posts = calculate_impact_scores(video_linked_posts)

        # Transcribe audio from videos
        transcribed_posts = self.transcribe(scored_posts)

        # Extract hook
        posts_with_hook = self.add_hook(transcribed_posts)

        # Extract visual features
        posts_with_visual_features = self.extract_visual_features(posts_with_hook)

        # Extract voice script and on-screen elements

        # Extract voice features

        # Extract background music features

        # Extract 3rd party footage features
        return posts_with_visual_features

    def download_videos(self, df: DataFrame) -> DataFrame:
        self.scraper.open_browser()

        for index, row in df.iterrows():
            url = row['url']
            post_id = row['post_id']
            vid_link = self._get_s3_video_link(url, post_id)
            df.at[index, "video_link"] = vid_link

        self.scraper.close_browser()
        return df

    def transcribe(self, df: DataFrame) -> DataFrame:
        df["full_script"] = df["video_link"].apply(self._transcribe_video)
        return df

    def add_hook(self, df: DataFrame) -> DataFrame:
        df = df.apply(self._get_hook, axis=1)
        return df

    def extract_visual_features(self, df: DataFrame) -> DataFrame:
        df = df.apply(self._extract_visual_features, axis=1)
        return df

    """
        Helper functions
    """

    def _get_s3_video_link(self, video_url: str, post_id: str) -> str | None:
        filename = f"tiktok_{post_id}.mp4"

        if self.s3.exists_in_bucket(self.video_bucket, filename):
            print(f"Video already exists: {filename}")
            return f"s3://{self.video_bucket}/{filename}"

        temp_file = self.scraper.download_video(video_url, filename)
        if temp_file is None:
            return None
        video_link = self.s3.upload_to_s3(self.video_bucket, filename, temp_file)
        os.remove(temp_file)

        # Wait for 5-15 secs before downloading the next video
        time.sleep(random.randint(5, 15))
        return video_link

    def _transcribe_video(self, s3_url: str) -> str | None:
        video_filename = os.path.basename(s3_url)
        video_dir = tempfile.gettempdir()

        video_file_path = f"{video_dir}/{video_filename}"
        audio_file_path = f"{video_dir}/{os.path.splitext(video_filename)[0]}.wav"

        if not self.s3.download_from_s3(s3_url, video_file_path):
            return None

        if not extract_audio(video_file_path, audio_file_path):
            return None

        print("Transcribing audio...")
        transcription = self.audio.transcribe(audio_file_path)

        # cleanup
        os.remove(video_file_path)
        os.remove(audio_file_path)

        return transcription

    def _get_hook(self, row):
        s3_url = row["video_link"]
        full_script = row["full_script"]

        video_filename = os.path.basename(s3_url)
        video_dir = tempfile.gettempdir()
        video_file_path = f"{video_dir}/{video_filename}"

        if not self.s3.download_from_s3(s3_url, video_file_path):
            row["screen_hook"] = None
            row["audio_hook"] = None
            row["style"] = None
            return row

        frame = extract_hook_frame(video_file_path, frame_time=1)
        print("Calling AGENT to generate screen hook...")
        screen_hook = self.llm.generate_screen_hook(frame)
        audio_hook = get_audio_hook(full_script)
        print("Calling AGENT to analyze hook...")
        shooting_style = self.llm.generate_hook_analysis(frame, full_script)

        os.remove(video_file_path)

        row["screen_hook"] = screen_hook
        row["audio_hook"] = audio_hook
        row["style"] = shooting_style.__dict__

        return row

    def _extract_visual_features(self, row):
        # download video locally
        s3_url = row['video_link']
        video_filename = os.path.basename(s3_url)
        video_dir = tempfile.gettempdir()
        video_file_path = f"{video_dir}/{video_filename}"

        if not self.s3.download_from_s3(s3_url, video_file_path):
            row["visual"] = None
            return row

        video_duration = get_video_duration_cv2(video_file_path)
        keyframes = extract_keyframes(video_file_path, min(video_duration, 5.0))

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

        # cleanup
        os.remove(video_file_path)

        row["visual"] = visual_features
        return row
