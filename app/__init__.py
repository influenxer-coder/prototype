import atexit
import signal
import sys

import weaviate
from flask import Flask
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from weaviate.auth import Auth
from weaviate.client import WeaviateClient

from app.config.settings import Config

# Global variables
weaviate_client = None
selenium_driver = None


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize clients connections
    global weaviate_client
    global selenium_driver
    selenium_driver = connect_to_browser()
    weaviate_client = connect_weaviate_db()

    # Register shutdown functions
    atexit.register(shutdown_app)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    signal.signal(signal.SIGINT, handle_shutdown_signal)

    # Register blueprints
    from app.routes import recommendation_routes, ingestion_routes
    app.register_blueprint(recommendation_routes.bp)
    app.register_blueprint(ingestion_routes.bp)

    return app


def connect_weaviate_db() -> WeaviateClient:
    weaviate_url = Config.WEAVIATE_URL
    weaviate_api_key = Config.WEAVIATE_API_KEY
    openai_api_key = Config.OPENAI_API_KEY

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
        headers={'X-OpenAI-Api-key': openai_api_key}
    )
    print(f"DB Client Status: {"True" if client.is_ready() else "False"}")
    return client


def connect_to_browser():
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-webrtc")
    chrome_options.add_argument("--disable-3d-apis")
    chrome_options.add_argument("--enable-unsafe-swiftshader")
    chrome_options.add_argument("--log-level=3")  # Suppress logs

    # Initialize the WebDriver
    driver = webdriver.Chrome(service=Service(), options=chrome_options)

    # Enable network logging
    driver.execute_cdp_cmd("Network.enable", {})
    return driver


def shutdown_app():
    """Perform cleanup when app shuts down"""
    print("Application shutting down, cleaning up resources...")
    close_weaviate_connection()
    close_selenium_driver()


def close_weaviate_connection():
    global weaviate_client
    if weaviate_client:
        print("Cleaning up Weaviate connections")
        weaviate_client.close()


def close_selenium_driver():
    global selenium_driver
    if selenium_driver and selenium_driver.service.is_connectable():
        print("Closing selenium browser")
        selenium_driver.quit()


def handle_shutdown_signal(sig, frame):
    """Handle termination signals"""
    print(f"Received shutdown signal: {sig}")
    shutdown_app()
    sys.exit(0)
