import os
import tempfile

import requests
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class ScraperService:
    def __init__(self, driver):
        self.driver = driver

    def download_video(self, video_url: str, filename: str) -> str | None:
        temp_dir = tempfile.gettempdir()
        download_path = os.path.join(temp_dir, filename)

        try:
            # Navigate to the TikTok video URL
            self.driver.get(video_url)
            print(f"Opened TikTok video page: {video_url}")

            # Wait for the main content div to load
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.ID, "main-content-video_detail"))
            )
            print("Main content div loaded.")

            # Locate the <video> tag
            video_element = self.driver.find_element(By.CSS_SELECTOR, "#main-content-video_detail video")
            print("Found <video> tag.")

            # Extract <source> tags inside the <video> tag
            source_tags = video_element.find_elements(By.TAG_NAME, "source")
            if not source_tags:
                print("No <source> tags found inside the <video> tag.")
                return None

            # Get the video URL from the first <source> tag
            video_src = source_tags[0].get_attribute("src")
            if not video_src:
                print("No src attribute found in <source> tag.")
                return None

            print(f"Video URL extracted: {video_src}")

            # Extract cookies from Selenium
            cookies = self.driver.get_cookies()
            print("Extracted cookies from Selenium session.")

            # Use the cookies in a requests session
            session = requests.Session()

            # Add cookies to the session
            for cookie in cookies:
                session.cookies.set(cookie["name"], cookie["value"])

            # Download the video using requests
            response = session.get(video_src, stream=True)
            if response.status_code == 200:
                try:
                    with open(download_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            f.write(chunk)
                    print(f"Download completed. Video saved as: {download_path}")
                    return download_path
                except Exception as e:
                    print(f"Error writing file: {str(e)}")
                    return None
            else:
                print(f"Failed to download video. HTTP Status Code: {response.status_code}")
                return None
        except Exception as e:
            print(f"An error occurred while downloading the video {video_url}: {str(e)}")
            return None
