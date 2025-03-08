import json
from typing import Dict, List, Optional

import numpy as np
import requests

from app.config.settings import Config
from app.models.video import KeyframeContext
from app.models.video import ShootingStyle
from app.utils.prompt import load_prompt, extract_json
from app.utils.video import frame_to_base64


# TODO: Interface class with actual classes for Chat GPT, Claude AI, Perplexity
class LlmAgentService:

    def _format_hook_details(self, analysis_text: str) -> Dict:
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

    def _generate_response(self, content, model=Config.MODEL.CLAUDE_3_HAIKU.value):
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
                raise Exception(f"API call failed: {response.text}")

            return response.json()["content"][0]["text"]
        except Exception as e:
            raise e

    def _get_response_from_agent(self, content, json_response=True, model=Config.MODEL.CLAUDE_3_HAIKU.value):
        while True:
            try:
                raw_response = self._generate_response(content, model)
                if json_response:
                    return extract_json(raw_response)
                return raw_response
            except json.JSONDecodeError:
                print(f"Retrying response generation")
            except Exception as e:
                raise e

    def generate_summary(self, keyframes: List[KeyframeContext], caption: str):
        """Send keyframes and audio to Claude for analysis."""
        try:
            # Load and format the prompt
            prompt_template = load_prompt('summary_generator')
            post_caption = caption if caption is not None else ""
            prompt = prompt_template.replace("{caption}", post_caption)

            content = [{
                "type": "text",
                "text": prompt
            }]

            # Add each moment as input
            for i, kf in enumerate(keyframes, 1):
                content.append({
                    "type": "text",
                    "text": f"""
    === Moment {kf.frame_number} ===
    Timestamp: {kf.timestamp:.2f} seconds
    """
                })
                content.append(frame_to_base64(kf.image))
                content.append({
                    "type": "text",
                    "text": f"""
    Audio from {kf.window_start:.2f}s to {kf.window_end:.2f}s:
    {kf.audio_transcript}
    -------------------"""
                })

            try:
                analysis_data = self._get_response_from_agent(content)

                summary = analysis_data["summary"]

                return summary

            except Exception as e:
                print(f"Error during API call: {str(e)}")

        except Exception as e:
            print(f"Error in generate_summary: {str(e)}")

    def generate_screenplay(self, summary, complete_transcript) -> dict:
        """
        Generate screenplay from video analysis and complete transcript.
        """
        try:
            prompt_template = load_prompt('screenplay_generator')

            content = [{
                "type": "text",
                "text": prompt_template
            }, {
                "type": "text",
                "text": f"""
    === Video Analysis ===
    {json.dumps(summary, indent=2)}

    === Complete Transcript ===
    {complete_transcript}
    """
            }]

            try:
                screenplay_data = self._get_response_from_agent(content)
                return screenplay_data

            except Exception as e:
                print(f"Error during API call: {str(e)}")
                return {}

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

            # Make sure it's not too long
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
        if frame is None:
            return ShootingStyle(
                visual_style_summary="Error: Could not extract frame from video.",
                visual_style="Error: Could not extract frame from video.",
                audio_style="Error: Could not analyze audio style.",
                creator_instructions="Error: Could not generate instructions."
            )

        visual_style_summary = self.generate_visual_style(frame)

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
            formatted_response = self._format_hook_details(response)
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

            response = self._get_response_from_agent(content, model=Config.MODEL.CLAUDE_3_SONNET.value)
            return response

        except Exception as e:
            print(f"Error in generate_visual_features: {str(e)}")
            return None

    def suggest_edits(self, comparison_request: dict) -> Optional[str]:

        try:
            prompt = load_prompt('edit_recommendations')

            content = [{
                "type": "text",
                "text": prompt
            }, {
                "type": "text",
                "text": json.dumps(comparison_request)
            }]
            response = self._get_response_from_agent(content, json_response=False,
                                                     model=Config.MODEL.CLAUDE_3_SONNET.value)

            return response

        except Exception as e:
            print(f"Error in suggest_edits: {str(e)}")
            return None
