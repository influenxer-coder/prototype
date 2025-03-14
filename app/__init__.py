import atexit
import signal
import sys

import weaviate
from flask import Flask
from weaviate.auth import Auth
from weaviate.client import WeaviateClient

from app.config.settings import Config

# Global variables
weaviate_client = None


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize clients connections
    global weaviate_client
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


def shutdown_app():
    """Perform cleanup when app shuts down"""
    print("Application shutting down, cleaning up resources...")
    global weaviate_client
    if weaviate_client:
        print("Cleaning up Weaviate connections")
        weaviate_client.close()


def handle_shutdown_signal(sig, frame):
    """Handle termination signals"""
    print(f"Received shutdown signal: {sig}")
    shutdown_app()
    sys.exit(0)
