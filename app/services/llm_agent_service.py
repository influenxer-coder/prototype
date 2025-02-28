import json
from typing import List

import requests

from app.config.settings import Config
from app.models.video import KeyframeAudioContext, VideoAnalysisSummary
from app.utils.prompt import load_prompt, extract_json
from app.utils.video import frame_to_base64


# TODO: Interface class with actual classes for Chat GPT, Claude AI, Perplexity
class LlmAgentService:
    def generate_summary(self, keyframes: List[KeyframeAudioContext], caption: str) -> VideoAnalysisSummary:
        """Send keyframes and audio to Claude for analysis."""
        try:
            # Load and format the prompt
            prompt_template = load_prompt('summary_generator', 'claude')
            prompt = prompt_template.replace("{caption}", caption)

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

                raw_response = response.json()["content"][0]["text"]

                print(raw_response)

                analysis_data = extract_json(raw_response)

                summary = analysis_data["summary"]

                return summary

            except Exception as e:
                print(f"Error during API call: {str(e)}")
                raise

        except Exception as e:
            print(f"Error in generate_summary: {str(e)}")
            raise

    def generate_screenplay(self, summary, complete_transcript) -> dict:
        """
        Generate screenplay from video analysis and complete transcript.
        """
        try:
            # Load and format the prompt
            prompt_template = load_prompt('screenplay_generator', 'claude')

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

                raw_response = response.json()["content"][0]["text"]

                print(raw_response)

                screenplay_data = extract_json(raw_response)

                return screenplay_data

            except Exception as e:
                print(f"Error during API call: {str(e)}")
                raise

        except Exception as e:
            print(f"Error in generate_screenplay: {str(e)}")
            raise
