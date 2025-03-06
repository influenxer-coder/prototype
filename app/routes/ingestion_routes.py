from flask import Blueprint, request, jsonify, Response

from app.services.ingestion_service import IngestionService
from app.utils.dataframe import get_dataframe, get_dict

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

    response = ingestion_service.process(posts)

    response = get_dict(response)
    return jsonify(response)
