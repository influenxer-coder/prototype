import json
from typing import List

import numpy as np
import requests

from app.config.settings import Config
from app.models.video import KeyframeAudioContext, VideoAnalysisSummary
from app.utils.prompt import load_prompt, extract_json
from app.utils.video import frame_to_base64


# TODO: Interface class with actual classes for Chat GPT, Claude AI, Perplexity
class LlmAgentService:
    def _generate_response(self, content):
        try:
            response = requests.post(
                Config.LLM_API_URL,
                headers={
                    "x-api-key": Config.LLM_API_KEY,
                    "anthropic-version": Config.LLM_API_VERSION,
                    "content-type": "application/json"
                },
                json={
                    "model": Config.MODEL_NAME,
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

    def _generate_json_response(self, content):
        while True:
            try:
                raw_response = self._generate_response(content)
                return extract_json(raw_response)
            except json.JSONDecodeError:
                print(f"Retrying response generation")
            except Exception as e:
                raise e

    def generate_summary(
            self,
            keyframes: List[KeyframeAudioContext],
            caption: str
    ) -> VideoAnalysisSummary:
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
                analysis_data = self._generate_json_response(content)

                summary = analysis_data["summary"]

                return summary

            except Exception as e:
                print(f"Error during API call: {str(e)}")
                return VideoAnalysisSummary(description="", key_moments=[])

        except Exception as e:
            print(f"Error in generate_summary: {str(e)}")
            return VideoAnalysisSummary(description="", key_moments=[])

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
                screenplay_data = self._generate_json_response(content)
                return screenplay_data

            except Exception as e:
                print(f"Error during API call: {str(e)}")
                return {}

        except Exception as e:
            print(f"Error in generate_screenplay: {str(e)}")
            return {}

    def generate_hook(self, frame: np.ndarray) -> str:
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
        return self._generate_response(content)
