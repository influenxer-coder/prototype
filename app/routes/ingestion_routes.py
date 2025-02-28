from flask import Blueprint, request, jsonify

from app.services.ingestion_service import IngestionService
from app.utils.dataframe import get_dataframe, get_dict, calculate_impact_scores

bp = Blueprint('ingestion_routes', __name__, url_prefix='/ingest')

ingestion_service = IngestionService()


@bp.route('/', methods=['POST'])
def ingest_records():
    data = request.json
    if not isinstance(data, list):
        data = [data]

    posts = get_dataframe(data)
    video_linked_posts = ingestion_service.download_vids(posts)
    scored_posts = calculate_impact_scores(video_linked_posts)

    response = get_dict(scored_posts)
    return jsonify(response)
