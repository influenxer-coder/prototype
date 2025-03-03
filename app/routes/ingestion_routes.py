from flask import Blueprint, request, jsonify, Response

from app.services.ingestion_service import IngestionService
from app.utils.dataframe import get_dataframe, get_dict, calculate_impact_scores

bp = Blueprint('ingestion_routes', __name__, url_prefix='/ingest')

ingestion_service = IngestionService()


@bp.route('/', methods=['POST'])
def ingest_records() -> Response:
    data = request.json
    if not isinstance(data, list):
        data = [data]

    posts = get_dataframe(data)

    # TODO: remove this line later
    posts = posts.head()

    video_linked_posts = ingestion_service.download_vids(posts)
    scored_posts = calculate_impact_scores(video_linked_posts)
    transcribed_posts = ingestion_service.transcribe(scored_posts)
    hooked_posts = ingestion_service.add_hook(transcribed_posts)

    response = get_dict(hooked_posts)
    return jsonify(response)
