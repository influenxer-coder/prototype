import os
import random
import tempfile
import time

from pandas.core.frame import DataFrame

from app.services.feature_extraction_service import FeatureExtractionService
from app.services.client.s3_service import S3Service
from app.services.client.scraper_service import ScraperService
from app.utils.audio import extract_audio
from app.utils.dataframe import calculate_impact_scores
from app.config.settings import Config


class IngestionService:
    def __init__(self):
        self.feature_extraction_service = FeatureExtractionService()
        self.s3 = S3Service()
        self.scraper = ScraperService()
        self.video_bucket = Config.AWS_S3_BUCKET

    def process(self, posts: DataFrame) -> DataFrame:
        # Download videos from TikTok
        video_linked_posts = self.download_videos(posts)

        # Calculate impact scores
        scored_posts = calculate_impact_scores(video_linked_posts)

        # Transcribe audio from videos
        transcribed_posts = self.transcribe(scored_posts)

        # Extract hook features
        posts_with_hook = self.add_hook(transcribed_posts)

        # Extract visual features
        posts_with_visual_features = self.extract_visual_features(posts_with_hook)

        # Extract keyframe features
        posts_with_style_features = self.extract_style_features(posts_with_visual_features)

        # Extract voice script and on-screen elements

        # Extract voice features

        # Extract background music features

        # Extract 3rd party footage features
        return posts_with_style_features

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
        return df.apply(self._extract_visual_features, axis=1)

    def extract_style_features(self, df: DataFrame) -> DataFrame:
        return df.apply(self._extract_style_features, axis=1)

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
        transcription = self.feature_extraction_service.transcribe(audio_file_path)

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

        visual_hook = self.feature_extraction_service.get_visual_hook(video_file_path, full_script)

        os.remove(video_file_path)

        row["screen_hook"] = visual_hook["screen_hook"]
        row["audio_hook"] = visual_hook["audio_hook"]
        row["style"] = visual_hook["shooting_style"].__dict__

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

        visual_features = self.feature_extraction_service.get_visual_features(video_file_path)

        # cleanup
        os.remove(video_file_path)

        row["visual"] = visual_features
        return row

    def _extract_style_features(self, row):
        s3_url = row["video_link"]

        video_filename = os.path.basename(s3_url)
        video_dir = tempfile.gettempdir()
        video_file_path = f"{video_dir}/{video_filename}"

        if not self.s3.download_from_s3(s3_url, video_file_path):
            row["creator_visible"] = None
            row["product_visible"] = None
            return row

        style_features = self.feature_extraction_service.get_style_features(video_file_path)

        row["creator_visible"] = style_features["creator_visible"]
        row["product_visible"] = style_features["product_visible"]

        return row
