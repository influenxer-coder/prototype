import json
import time
from typing import List, Optional, Dict

import numpy as np
import requests

from app.config.settings import Config
from app.models.video import KeyframeContext
from app.models.video import ShootingStyle
from app.utils.prompt import load_prompt, extract_json
from app.utils.video import frame_to_base64


class LlmAgentService:
    def __init__(self):
        self.base_model = Config.MODEL.CLAUDE_3_HAIKU.value

    def _generate_response(self, content, model=None):
        model = self.base_model if not model else model
        while True:
            try:
                response = requests.post(
                    Config.LLM_API_URL,
                    headers={
                        "x-api-key": Config.LLM_API_KEY,
                        "anthropic-version": Config.LLM_API_VERSION,
                        "content-type": "application/json"
                    },
                    json={
                        "model": model,
                        "max_tokens": Config.MAX_TOKENS,
                        "messages": [{
                            "role": "user",
                            "content": content
                        }]
                    }
                )

                if response.status_code != 200:
                    res = response.json()
                    if res['type'] == "error" and res['error']['type'] == "rate_limit_error":
                        time.sleep(60)
                        continue

                    raise Exception(res)

                return response.json()["content"][0]["text"]
            except Exception as e:
                raise e

    def _generate_json_response(self, content, model=None):
        while True:
            try:
                raw_response = self._generate_response(content, model)
                return extract_json(raw_response)
            except json.JSONDecodeError:
                print(f"Retrying JSON response generation...")
            except Exception as e:
                raise e

    def generate_summary(self, keyframes: List[KeyframeContext], caption: str):
        """Send keyframes and audio to Claude for analysis."""

        def create_moment_header(keyframe) -> dict:
            """Create the header text for a moment."""
            return {
                "type": "text",
                "text": f"=== Moment {keyframe.frame_number} ===\n"
                        f"Timestamp: {keyframe.timestamp:.2f} seconds\n"
            }

        def create_audio_transcript(keyframe) -> dict:
            """Create the audio transcript text for a moment."""
            return {
                "type": "text",
                "text": f"Audio from {keyframe.window_start:.2f}s to {keyframe.window_end:.2f}s:\n"
                        f"{keyframe.audio_transcript}\n"
                        f"-------------------"
            }

        try:
            # Load and format the prompt
            prompt_template = load_prompt('summary_generator')
            post_caption = caption if caption is not None else ""
            prompt = prompt_template.replace("{caption}", post_caption)

            content = [{"type": "text", "text": prompt}]
            for kf in keyframes:
                content.extend([
                    create_moment_header(kf),
                    frame_to_base64(kf.image),
                    create_audio_transcript(kf)
                ])

            analysis_data = self._generate_json_response(content)
            summary = analysis_data["summary"]
            return summary
        except Exception as e:
            print(f"Error in generate_summary: {str(e)}")
            return {}

    def generate_screenplay(self, summary, complete_transcript) -> dict:
        """
        Generate screenplay from video analysis and complete transcript.
        """
        try:
            prompt_template = load_prompt('screenplay_generator', provider='recommendation')

            content = [
                {"type": "text", "text": prompt_template},
                {"type": "text",
                 "text": f"=== Video Analysis ===\n"
                         f"{json.dumps(summary, indent=2)}\n\n"
                         f"=== Complete Transcript ===\n"
                         f"{complete_transcript}"}
            ]

            screenplay_data = self._generate_json_response(content)
            return screenplay_data
        except Exception as e:
            print(f"Error in generate_screenplay: {str(e)}")
            return {}

    def generate_screen_hook(self, frame: np.ndarray) -> str:
        """
        Extract the hook text from a video frame using Claude AI

        Args:
            frame: Frame in numpy array

        Returns:
            str: Extracted hook text or error message
        """
        if frame is None:
            return "Error: Could not extract frame from video."

        base64_image = frame_to_base64(frame)
        content = [{
            "type": "text",
            "text": "This is a frame from a video. Please identify and extract the main caption or hook text that "
                    "appears on the screen. Focus only on the text that appears to be the main attention-grabbing "
                    "statement or hook. If there is no hook, return 'NO HOOK'."
                    "Return just the text without any additional commentary."
        }, base64_image]

        try:
            response = self._generate_response(content)
            if response == 'NO HOOK':
                return 'No caption text detected on screen.'
            return response
        except Exception as e:
            print(f"API error in caption extraction: {e}")
            return f"Error: Caption extraction failed"

    def generate_visual_style(self, frame: np.ndarray) -> str:
        """
        Get a concise 10-15 word description of what the creator is doing

        Args:
            frame (np.ndarray): Video image frame

        Returns:
            str: Concise description of the visual style
        """
        if frame is None:
            return "Error: Could not extract frame from video."

        prompt = load_prompt("visual_style_generator")

        base64_image = frame_to_base64(frame)
        content = [{
            "type": "text",
            "text": prompt
        }, base64_image]

        try:
            response = self._generate_response(content)
            words = response.split()
            if len(words) > 15:
                response = ' '.join(words[:15])
            return response
        except Exception as e:
            print(f"API error in visual style generation: {e}")
            return f"Error: Visual Style generation failed"

    def generate_hook_analysis(self, frame: np.ndarray, transcript: str = "") -> ShootingStyle:
        """
        Analyze the given frame and transcript to describe what the creator is doing and provide actionable instructions

        Args:
            frame (np.ndarray): Image frame from the TikTok video
            transcript (str): The video transcript to enhance style analysis

        Returns:
            ShootingStyle: Detailed style information with actionable instructions
        """

        def format_hook_details(analysis_text: str) -> Dict:
            creator_instructions = ""

            if "VISUAL_STYLE:" in analysis_text and "AUDIO_STYLE:" in analysis_text:
                parts = analysis_text.split("AUDIO_STYLE:")
                visual_part = parts[0].split("VISUAL_STYLE:")[1].strip()
                visual_style = visual_part

                if "CREATOR_INSTRUCTIONS:" in parts[1]:
                    remaining = parts[1].split("CREATOR_INSTRUCTIONS:")
                    audio_style = remaining[0].strip()
                    creator_instructions = remaining[1].strip()
                else:
                    audio_style = parts[1].strip()
            else:
                # Fallback if the format is different
                visual_style = "Could not parse visual style from analysis."
                audio_style = "Could not parse audio style from analysis."
                creator_instructions = analysis_text  # Just use the full text as instructions

            return {
                "visual_style": visual_style,
                "audio_style": audio_style,
                "creator_instructions": creator_instructions
            }

        if frame is None:
            return ShootingStyle(
                visual_style_summary="Error: Could not extract frame from video.",
                visual_style="Error: Could not extract frame from video.",
                audio_style="Error: Could not analyze audio style.",
                creator_instructions="Error: Could not generate instructions."
            )

        visual_style_summary = self.generate_visual_style(frame)

        """
        We should remove the below code. 
        We are asking LLM to generate something without providing much context.
        [hook_analysis_generator] prompt
        """

        transcript = transcript if transcript else ''
        transcript = transcript if len(transcript) <= 500 else transcript[:500] + '...'

        prompt_template = load_prompt('hook_analysis_generator')
        prompt = prompt_template.replace("{transcript}", transcript)

        base64_image = frame_to_base64(frame)
        content = [{
            "type": "text",
            "text": prompt
        }, base64_image]

        try:
            response = self._generate_response(content)
            formatted_response = format_hook_details(response)
            return ShootingStyle(
                visual_style_summary=visual_style_summary,
                visual_style=formatted_response['visual_style'],
                audio_style=formatted_response['audio_style'],
                creator_instructions=formatted_response['creator_instructions']
            )
        except Exception as e:
            print(f"API error in scene analysis: {e}")
            return ShootingStyle(
                visual_style_summary=visual_style_summary,
                visual_style="Error: Scene analysis failed",
                audio_style="Error: Could not analyze audio style",
                creator_instructions="Error: Could not generate instructions"
            )

    def generate_visual_features(self, keyframes: List[KeyframeContext]):
        try:
            prompt = load_prompt('visual_feature_extractor')

            content = [{"type": "text", "text": prompt}]

            # Add each moment as input
            for kf in keyframes:
                content.extend([
                    {"type": "text", "text": f"\n=== Moment {kf.frame_number} ===\n"},
                    frame_to_base64(kf.image)
                ])

            response = self._generate_json_response(content, model=Config.MODEL.CLAUDE_3_SONNET.value)
            return response

        except Exception as e:
            print(f"Error in generate_visual_features: {str(e)}")
            return None

    def suggest_edits(self, comparison_request: dict) -> Optional[str]:

        try:
            prompt = load_prompt('edit_recommendations', provider='recommendation')

            content = [{
                "type": "text",
                "text": prompt
            }, {
                "type": "text",
                "text": json.dumps(comparison_request)
            }]

            response = self._generate_response(content, model=Config.MODEL.CLAUDE_3_SONNET.value)
            return response
        except Exception as e:
            print(f"Error in suggest_edits: {str(e)}")
            return None

    def generate_style_features(self, keyframes: List[tuple]) -> dict:
        """
        Analyze keyframes to describe video properties

        Args:
            keyframes (List[tuple]): List of the critical frames from the video (frame_number, frame_time, frame)

        Returns:
            dict: Properties identified by analyzing video keyframes
        """
        if keyframes is None:
            return {
                "creator_visible": None,
                "product_visible": None
            }

        prompt = load_prompt('style_feature_extractor')

        image_contents = [frame_to_base64(keyframe[2]) for keyframe in keyframes]

        chunks = [image_contents[x:x + 5] for x in range(0, len(image_contents), 5)]

        face_visible = hand_visible = product_visible = False

        for chunk in chunks:
            content = [{
                "type": "text",
                "text": prompt
            }]
            content.extend(chunk)

            try:
                response = self._generate_json_response(content, model=Config.MODEL.CLAUDE_3_5_HAIKU.value)
                face_visible = face_visible or response['face_visible']
                hand_visible = hand_visible or response['hand_visible']
                product_visible = product_visible or response['product_visible']
                if face_visible and product_visible:
                    break
            except Exception as e:
                print(f"API error in keyframe analysis: {e}")
                return {
                    "creator_visible": None,
                    "product_visible": None
                }

        return {
            'creator_visible': "Face is visible" if face_visible else ("Only hands" if hand_visible else "No"),
            'product_visible': product_visible
        }
