import json
import requests
from typing import List
from app.config.settings import Config
from app.models.video import KeyframeAudioContext, VideoAnalysis
from app.utils.video_utils import frame_to_base64, extract_json
from app.utils.prompt_loader import PromptLoader


class LlmAgentService:
    def __init__(self):
        self.prompt_loader = PromptLoader()

    def generate_summary(self, keyframes: List[KeyframeAudioContext], caption: str) -> VideoAnalysis:
        """Send keyframes and audio to Claude for analysis."""
        try:
            # Load and format the prompt
            prompt_template = self.prompt_loader.load_prompt('summary_generation')
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
                analysis_data = extract_json(raw_response)

                return VideoAnalysis(
                    summary=analysis_data["summary"],
                    key_moments=analysis_data["key_moments"]
                )

            except Exception as e:
                print(f"Error during API call: {str(e)}")
                raise

        except Exception as e:
            print(f"Error in generate_summary: {str(e)}")
            raise
