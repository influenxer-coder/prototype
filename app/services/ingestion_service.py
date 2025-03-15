import os
import random
import tempfile
import time
from typing import List, Optional

from pandas.core.frame import DataFrame

from app import weaviate_client, selenium_driver
from app.config.settings import Config
from app.models import post as Post
from app.services.client.s3_service import S3Service
from app.services.client.scraper_service import ScraperService
from app.services.client.vector_db_service import VectorDBService
from app.services.feature_extraction_service import FeatureExtractionService
from app.utils.audio import extract_audio
from app.utils.dataframe import calculate_impact_scores, create_db_objects


class IngestionService:
    def __init__(self):
        self.feature_extraction_service = FeatureExtractionService()
        self.s3 = S3Service()
        self.vector_db = VectorDBService(weaviate_client)
        self.scraper = ScraperService(selenium_driver)
        self.video_bucket = Config.AWS_S3_BUCKET

    def process(self, posts: DataFrame) -> List[dict]:
        processed_batches = []

        batch_size = Config.BATCH_SIZE

        for start in range(0, len(posts), batch_size):
            end = start + batch_size
            batch = posts.iloc[start:end]

            processed_batch = (
                self.filter_records(batch)
                .pipe(self.download_videos)
                .pipe(calculate_impact_scores)
                .pipe(self.transcribe)
                .pipe(self.extract_style_features)
                .pipe(self.add_hook)
                .pipe(self.extract_visual_features)
                .pipe(self.extract_audio_features)
                .pipe(self.extract_shooting_style)
                .pipe(self.cleanup)
                .drop(
                    columns=[Config.LOCAL_VIDEO_PATH, Config.LOCAL_AUDIO_PATH, Config.LOCAL_SPEECH_PATH],
                    errors='ignore'
                )
            )
            saved_batch = self.add_to_vector_db(processed_batch)
            processed_batches.extend(saved_batch)

        return processed_batches

    def filter_records(self, df: DataFrame) -> DataFrame:
        df = df[~df['post_id'].apply(lambda x: self.vector_db.record_exists(Post.get_schema(), x))]
        return df

    def download_videos(self, df: DataFrame) -> DataFrame:
        for index, row in df.iterrows():
            url = row['url']
            post_id = row['post_id']
            s3_video_link, local_video_path = self._get_video_links(url, post_id)
            df.at[index, Config.S3_VIDEO_URL] = s3_video_link
            df.at[index, Config.LOCAL_VIDEO_PATH] = local_video_path

        return df

    def transcribe(self, df: DataFrame) -> DataFrame:
        df = df.apply(self._transcribe_video, axis=1)
        return df

    def add_hook(self, df: DataFrame) -> DataFrame:
        df = df.apply(self._get_hook, axis=1)
        return df

    def extract_visual_features(self, df: DataFrame) -> DataFrame:
        df = df.apply(self._extract_visual_features, axis=1)
        return df

    def extract_style_features(self, df: DataFrame) -> DataFrame:
        df = df.apply(self._extract_style_features, axis=1)
        return df

    def extract_audio_features(self, df: DataFrame) -> DataFrame:
        df = df.apply(self._extract_audio_features, axis=1)
        return df

    def extract_shooting_style(self, df: DataFrame) -> DataFrame:
        df = df.apply(self._extract_shooting_style, axis=1)
        return df

    def add_to_vector_db(self, df: DataFrame) -> Optional[List[dict]]:
        if not self.vector_db.create_collection(Post.get_schema()):
            return None
        try:
            posts = create_db_objects(df)
            if not self.vector_db.batch_add(Post.get_schema(), posts):
                return None
            return posts
        except Exception as e:
            print(f"Error: Unable to create object for the DB - {e}")
            return None

    def cleanup(self, df: DataFrame) -> DataFrame:
        df = df.apply(self._cleanup_local_files, axis=1)
        return df

    """
        Helper functions
    """

    def _cleanup_local_files(self, row):
        speech_file_path = row[Config.LOCAL_SPEECH_PATH]
        audio_file_path = row[Config.LOCAL_AUDIO_PATH]
        video_file_path = row[Config.LOCAL_VIDEO_PATH]

        if speech_file_path and os.path.exists(speech_file_path):
            os.remove(speech_file_path)
        if audio_file_path and os.path.exists(audio_file_path):
            os.remove(audio_file_path)
        if video_file_path and os.path.exists(video_file_path):
            os.remove(video_file_path)

        return row

    def _get_video_links(self, video_url: str, post_id: str) -> tuple[str | None, str | None]:
        """
        :param video_url:
        :param post_id:
        :return:
        tuple[s3 video link, local video link]
        """
        filename = f"tiktok_{post_id}.mp4"

        temp_dir = tempfile.gettempdir()
        local_path = os.path.join(temp_dir, filename)

        if self.s3.exists_in_bucket(self.video_bucket, filename):
            print(f"Video already exists in S3: {filename}")
            s3_link = f"s3://{self.video_bucket}/{filename}"
            if self.s3.download_from_s3(s3_link, local_path):
                return s3_link, local_path
            else:
                return s3_link, None
        else:
            temp_file = self.scraper.download_video(video_url, filename)
            if temp_file is None:
                return None, None

            s3_link = self.s3.upload_to_s3(self.video_bucket, filename, temp_file)
            # os.remove(temp_file)

            # Wait for 5-15 secs before downloading the next video
            time.sleep(random.randint(5, 15))
            return s3_link, temp_file

    def _transcribe_video(self, row):

        video_file_path = row[Config.LOCAL_VIDEO_PATH]
        video_dir = os.path.dirname(video_file_path)
        video_filename = os.path.basename(video_file_path)

        audio_file_path = f"{os.path.join(video_dir, os.path.splitext(video_filename)[0])}.wav"

        if not extract_audio(video_file_path, audio_file_path):
            row[Config.TRANSCRIPT] = None
            row[Config.LOCAL_AUDIO_PATH] = None
            return row

        print("Transcribing audio...")
        transcription = self.feature_extraction_service.transcribe(audio_file_path)

        row[Config.TRANSCRIPT] = transcription
        row[Config.LOCAL_AUDIO_PATH] = audio_file_path

        return row

    def _get_hook(self, row):
        video_file_path = row[Config.LOCAL_VIDEO_PATH]
        full_script = row[Config.TRANSCRIPT]

        hook = self.feature_extraction_service.get_audio_visual_hook(video_file_path, full_script)

        row[Config.HOOK] = hook

        return row

    def _extract_visual_features(self, row):
        video_file_path = row[Config.LOCAL_VIDEO_PATH]

        visual_features = self.feature_extraction_service.get_visual_features(video_file_path)

        row[Config.VISUAL] = visual_features
        return row

    def _extract_style_features(self, row):
        video_file_path = row[Config.LOCAL_VIDEO_PATH]
        transcript = row[Config.TRANSCRIPT]

        style_features = self.feature_extraction_service.get_style_features(video_file_path, transcript)

        row[Config.STYLE] = style_features

        return row

    def _extract_audio_features(self, row):
        audio_file_path = row[Config.LOCAL_AUDIO_PATH]

        if audio_file_path is None:
            row[Config.LOCAL_SPEECH_PATH] = None
            row[Config.AUDIO] = None
            return row

        print(f"Generating Audio features...")
        speech_audio_path = self.feature_extraction_service.isolate_speech(audio_file_path)

        if speech_audio_path is None:
            row[Config.LOCAL_SPEECH_PATH] = None
            row[Config.AUDIO] = None
            return row

        audio_features = self.feature_extraction_service.get_audio_features(speech_audio_path)

        row[Config.LOCAL_SPEECH_PATH] = speech_audio_path
        row[Config.AUDIO] = audio_features
        return row

    def _extract_shooting_style(self, row):
        style = row[Config.STYLE]
        full_script = row[Config.TRANSCRIPT]

        shooting_style = self.feature_extraction_service.get_shooting_style(style, full_script)
        row[Config.SHOOTING_STYLE] = shooting_style
        return row
