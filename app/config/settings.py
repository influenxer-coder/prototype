import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev')
    DATABASE_URL = os.getenv('DATABASE_URL')
    TIKTOK_API_KEY = os.getenv('TIKTOK_API_KEY')