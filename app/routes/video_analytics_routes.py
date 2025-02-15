from flask import Blueprint, request, jsonify
from app.services.video_analytics_service import VideoAnalyticsService
import os

bp = Blueprint('video', __name__)
video_analytics_service = VideoAnalyticsService()


@bp.route('/analyze_video', methods=['POST'])
def analyze_video():
    try:
        # Get video path from request
        data = request.json
        video_path, caption = data.get('url'), data.get('description')

        # in real use case we need to download video from S3 before processing it

        if not video_path:
            return jsonify({'error': 'No video path provided'}), 400

        if not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found at specified path'}), 404

        # Process the video directly from the path
        analysis = video_analytics_service.process_video(video_path, caption)

        return jsonify({
            'summary': analysis.summary,
            'key_moments': analysis.key_moments,
            'marketing_analysis': analysis.marketing_analysis,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500