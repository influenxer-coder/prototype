from flask import Flask
from app.config.settings import Config


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions

    # Register blueprints
    from app.routes import video_analytics_routes
    app.register_blueprint(video_analytics_routes.bp)

    return app