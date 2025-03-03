import os
import random
import tempfile
import time

from pandas.core.frame import DataFrame

from app.services.audio_service import AudioService
from app.services.llm_agent_service import LlmAgentService
from app.services.s3_service import S3Service
from app.services.scraper_service import ScraperService
from app.utils.audio import extract_audio
from app.utils.video import extract_hook_frame


class IngestionService:
    def __init__(self):
        self.s3 = S3Service()
        self.scraper = ScraperService()
        self.audio = AudioService()
        self.llm = LlmAgentService()
        self.video_bucket = "tapestry-tiktok-videos"

    def download_vids(self, df: DataFrame) -> DataFrame:
        self.scraper.open_browser()

        for index, row in df.iterrows():
            url = row['url']
            post_id = row['post_id']
            vid_link = self._get_s3_video_link(url, post_id)
            df.at[index, "video_link"] = vid_link

        self.scraper.close_browser()
        return df

    def transcribe(self, df: DataFrame) -> DataFrame:
        df['audio_transcription'] = df['video_link'].apply(self._transcribe_video)
        return df

    def add_hook(self, df: DataFrame) -> DataFrame:
        df = df.apply(self._get_hook, axis=1)
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

        transcription = self.audio.transcribe(audio_file_path)

        os.remove(video_file_path)
        os.remove(audio_file_path)

        print(f"Transcript: \n{transcription}")
        return transcription

    def _get_hook(self, row):
        s3_url = row["video_link"]

        video_filename = os.path.basename(s3_url)
        video_dir = tempfile.gettempdir()
        video_file_path = f"{video_dir}/{video_filename}"

        if not self.s3.download_from_s3(s3_url, video_file_path):
            row["visual_hook"] = None
            return row

        img = extract_hook_frame(video_file_path)
        visual_hook = self.llm.generate_hook(img)
        os.remove(video_file_path)

        print("-" * 80)
        print(f"Hook: {visual_hook}")
        print("-" * 80)

        row["visual_hook"] = visual_hook
        return row
