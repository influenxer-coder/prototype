import os
from enum import Enum

from dotenv import load_dotenv

load_dotenv()


class Model(Enum):
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    CLAUDE_3_SONNET = "claude-3-5-sonnet-20240620"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"


class Config:
    LLM_API_KEY = os.getenv('CLAUDE_API_KEY')
    LLM_API_URL = "https://api.anthropic.com/v1/messages"

    MODEL = Model

    MAX_TOKENS = 1500

    # Video processing settings
    MIN_SCENE_CHANGE_THRESHOLD = 15.0
    MIN_INTERVAL_SECONDS = 1.0

    # Audio processing settings
    AUDIO_MODELS = [
        "whisper",
        "google"
    ]

    # API versions
    LLM_API_VERSION = "2023-06-01"

    # AWS settings
    AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
    AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
    AWS_REGION = "us-east-2"
    AWS_S3_BUCKET = "tapestry-tiktok-videos"

    # Scoring Settings
    WEIGHTS = {
        "digg_count": 0.2,  # Likes
        "comment_count": 0.3,  # Comments
        "share_count": 0.2,  # Shares
        "play_count": 0.2,  # Views
        "recentness": 0.1  # Recentness
    }

    # Mini batch size for processing videos
    BATCH_SIZE = 10

    # Dataframe constants
    LOCAL_VIDEO_PATH = "local_video_path"
    LOCAL_AUDIO_PATH = "local_audio_path"
    S3_VIDEO_URL = "s3_video_url"
    TRANSCRIPT = "transcript"
    STYLE = "style"
    HOOK = "hook"
    VISUAL = "visual"
    AUDIO = "audio"
