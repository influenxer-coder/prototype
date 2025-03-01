import os

from flask import Blueprint, request, jsonify

from app.services.video_service import VideoService

bp = Blueprint('video', __name__)
video_analytics_service = VideoService()


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
        summary, screenplay = video_analytics_service.process_video(video_path, caption)

        return jsonify({
            'summary': summary,
            'screenplay': screenplay['screenplay'],
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
