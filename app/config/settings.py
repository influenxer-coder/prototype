import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    LLM_API_KEY = os.getenv('CLAUDE_API_KEY')
    LLM_API_URL = "https://api.anthropic.com/v1/messages"
    MODEL_NAME = "claude-3-haiku-20240307"
    MAX_TOKENS = 1500

    # Video processing settings
    MIN_SCENE_CHANGE_THRESHOLD = 20.0
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

    # Scoring Settings
    WEIGHTS = {
        "digg_count": 0.2,  # Likes
        "comment_count": 0.3,  # Comments
        "share_count": 0.2,  # Shares
        "play_count": 0.2,  # Views
        "recentness": 0.1  # Recentness
    }
