from flask import Flask

from app.config.settings import Config
from app.routes import recommendation_routes, ingestion_routes


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Register blueprints
    app.register_blueprint(recommendation_routes.bp)
    app.register_blueprint(ingestion_routes.bp)

    return app
