import os
import random
import time

from pandas.core.frame import DataFrame

from app.services.s3_service import S3Service
from app.services.scraper_service import ScraperService


class IngestionService:
    def __init__(self):
        self.s3 = S3Service()
        self.scraper = ScraperService()
        self.video_bucket = "tapestry-tiktok-videos"

    def download_vids(self, df: DataFrame) -> DataFrame:
        self.scraper.open_browser()

        for index, row in df.iterrows():
            url = row['url']
            post_id = row['post_id']
            vid_link = self.get_s3_video_link(url, post_id)
            df.at[index, "video_link"] = vid_link

        self.scraper.close_browser()
        return df

    """
        Helper functions
    """

    def get_s3_video_link(self, video_url: str, post_id: str):
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
