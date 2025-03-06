import os
from typing import List

from flask import Blueprint, request, jsonify

from app.models.video import Video
from app.services.recommendation_service import RecommendationService

bp = Blueprint('video', __name__)
recommendation_service = RecommendationService()


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
        summary, screenplay = recommendation_service.process_video(video_path, caption)

        return jsonify({
            'summary': summary,
            'screenplay': screenplay['screenplay'],
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/suggest_edits', methods=['POST'])
def suggest_edits():
    try:
        # get high performing videos
        data = request.json

        high_performing_videos: List[Video] = [Video(**item) for item in data.get('high_performing', [])]
        low_performing_video = Video(**data.get('low_performing'))

        """
        In real use case, we need to:
        1. extract the features of the low performing video
        2. get the features of the high performing videos using similarity search
        2. then perform comparison
        """

        edits = recommendation_service.suggest_edits(high_performing_videos, low_performing_video)

        return jsonify(edits)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
