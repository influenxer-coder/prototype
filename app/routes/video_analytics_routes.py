from flask import Blueprint, request, jsonify
from app.services.video_service import VideoAnalyticsService
from app.models.video import VideoRequest

bp = Blueprint('video-analytics', __name__)
analytics_service = VideoAnalyticsService()


@bp.route('/add_video', methods=['POST'])
def add_video():
    try:
        video_data = VideoRequest(**request.json)
        result = analytics_service.process_video(video_data)
        return jsonify(result), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/suggest_themes', methods=['GET'])
def suggest_themes():
    try:
        product_url = request.json["url"]
        result = analytics_service.suggest_themes(product_url)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'error': 'Failed to retrieve themes',
            'details': str(e)
        }), 500


@bp.route('/get_screenplay/<theme_id>', methods=['GET'])
def get_videos(theme_id):
    try:
        result = analytics_service.get_screenplay(theme_id)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'error': 'Failed to retrieve videos',
            'details': str(e)
        }), 500
