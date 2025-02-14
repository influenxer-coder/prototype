from app.models.video import VideoRequest


class VideoAnalyticsService:
    def __init__(self):
        # TODO: we need to replace this with actual DB
        self.videos_db = {}

    def process_video(self, video_data: VideoRequest):

        # need to download video

        # then perform analysis on the video (extract: summary, screenplay, video style)

        # add video analytics data to relevant DB

        return None

    def suggest_themes(self, product_url):

        # look up related videos in DB

        # perform clustering on the fetched videos based on video style

        # rank the videos within every cluster

        # suggest themes by taking the summary, screenplay and video style for every cluster

        return None

    def get_screenplay(self, theme_id):

        # look up DB for the theme_id and return the screenplay for the tagged videos

        return None