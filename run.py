import io
import sys

from flask_cors import CORS

from app import create_app
from app.services.scraper_service import ScraperService

app = create_app()

CORS(app)


@app.teardown_appcontext
def shutdown_session(exceptions=None):
    print("Closing open connections...")
    scraper = ScraperService()
    scraper.close_browser()


if __name__ == '__main__':
    # Set the console encoding to UTF-8
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    app.run(debug=True)
