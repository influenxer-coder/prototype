from dataclasses import dataclass
from datetime import datetime
from typing import Dict
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, HttpUrl


class TaggedUser(BaseModel):
    user_id: str
    user_handle: str
    user_url: HttpUrl
    user_name: str


class Music(BaseModel):
    authorname: str
    covermedium: HttpUrl
    id: str
    original: bool
    playurl: HttpUrl
    title: str


class DiscoveryInput(BaseModel):
    search_keyword: str
    what_to_collect: str = ""
    country: str = ""


class Input(BaseModel):
    url: HttpUrl
    discovery_input: DiscoveryInput


class VideoRequest(BaseModel):
    # Basic visual information
    url: HttpUrl
    post_id: str
    description: str
    create_time: datetime

    # Engagement metrics
    digg_count: int
    share_count: int
    collect_count: int
    comment_count: int
    play_count: int
    video_duration: int

    # Content metadata
    hashtags: List[str]
    original_sound: str
    post_type: str = "visual"
    width: int
    ratio: str

    # URLs and media
    video_url: HttpUrl
    preview_image: HttpUrl

    # Profile information
    profile_id: str
    profile_username: str
    profile_url: HttpUrl
    profile_avatar: HttpUrl
    profile_biography: Optional[str]
    profile_followers: int
    is_verified: bool = False

    # Music information
    music: Music

    # Additional metadata
    secu_id: str
    shortcode: str
    region: str
    tagged_user: Optional[List[TaggedUser]]
    tt_chain_token: str

    # Discovery information
    discovery_input: DiscoveryInput
    input: Input

    # Scoring (optional)
    recentness_score: Optional[float] = None
    raw_score: Optional[float] = None
    normalized_score: Optional[float] = None

    # Optional fields
    offical_item: Optional[bool] = False
    original_item: Optional[bool] = False
    cdn_url: Optional[str] = None
    commerce_info: Optional[Dict] = None
    carousel_images: Optional[List] = None


@dataclass
class KeyframeContext:
    frame_number: int
    timestamp: float
    image: np.ndarray
    audio_transcript: Optional[str]
    window_start: float
    window_end: float


@dataclass
class ShootingStyle:
    visual_style_summary: str
    visual_style: str
    audio_style: str
    creator_instructions: str


@dataclass
class VideoAnalysisSummary:
    description: str
    key_moments: List[dict]


class Feature(BaseModel):
    description: str
    score: int


class Subject(BaseModel):
    appearance: Feature
    contrast_with_background: Feature
    camera_proximity: Feature
    expressiveness: Feature


class Background(BaseModel):
    appeal: Feature
    distracting_elements: Feature
    lighting_quality: Feature


class TextOverlay(BaseModel):
    presence: Feature
    main_text: Feature


class VisualFeatures(BaseModel):
    subject: Subject
    background: Background
    text_overlay: TextOverlay


class Video(BaseModel):
    # url: str
    # post_id: int
    visual: VisualFeatures
