import json
from datetime import datetime
from typing import Dict, List

import pandas as pd
from pandas.core.frame import DataFrame

from app.config.settings import Config


def calculate_recentness_score(create_time):
    creation_date = datetime.fromisoformat(create_time.replace("Z", "+00:00"))
    today = datetime.now(creation_date.tzinfo)

    diff = (today - creation_date).days

    # Handle division by zero
    if diff == 0:
        return 365  # Max score

    recentness_score = 365 / diff
    return recentness_score


def calculate_impact_scores(df: DataFrame) -> DataFrame:
    df["recentness_score"] = df["create_time"].apply(calculate_recentness_score)
    df["impact_score"] = (
            df["digg_count"] * Config.WEIGHTS["digg_count"]
            + df["comment_count"] * Config.WEIGHTS["comment_count"]
            + df["share_count"] * Config.WEIGHTS["share_count"]
            + df["play_count"] * Config.WEIGHTS["play_count"]
            + df["recentness_score"] * Config.WEIGHTS["recentness"]
    )
    df = df.drop("recentness_score", axis=1)
    return df


def get_dataframe(data: List[Dict]) -> DataFrame:
    df = pd.DataFrame(data)
    return df


def get_dict(df: DataFrame) -> List[Dict]:
    return df.to_dict("records")


def create_db_objects(df: DataFrame) -> List[dict]:
    raw_posts = df.to_dict('records')
    posts = []
    for post in raw_posts:
        style = post['style']
        visual = post['visual']
        transcript = post["transcript"] if style and style["creator_speaking"] else ""
        post_obj = {
            "post_id": str(post["post_id"]),
            "url": post["url"],
            "description": post["description"],
            "impact_score": post["impact_score"],
            "search_term": post["search_term"],
            "transcript": transcript,
            "text_elements": "" if not visual else visual["text_overlay"]["main_text"]["description"],
            "shooting_style": post["shooting_style"],
            "object": json.dumps(post)
        }
        posts.append(post_obj)

    return posts
