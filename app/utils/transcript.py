import re


def get_audio_hook(full_script: str) -> str:
    if not full_script or full_script.startswith("Error:"):
        return "Error: No valid script to extract hook from."

    sentences = re.split(r'(?<=[.!?])\s+', full_script)
    if not sentences:
        return "Error: Could not identify sentences in the script."

    return sentences[0].strip()
