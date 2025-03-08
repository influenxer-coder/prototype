import json
from pathlib import Path


def load_prompt(prompt_name: str, provider: str = 'feature_extraction') -> str:
    """
    Load prompt from a text file.

    Args:
        provider: Name of the prompt provider
        prompt_name: Name of the prompt file without .txt extension

    Returns:
        str: Content of the prompt file
    """
    try:
        # Get the absolute path to the prompts directory
        provider_name = provider
        prompts_dir = Path(__file__).parent.parent / 'config' / 'prompts' / provider_name
        prompt_path = prompts_dir / f"{prompt_name}.txt"

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        with open(prompt_path, 'r', encoding='utf-8') as file:
            return file.read().strip()

    except Exception as e:
        raise Exception(f"Error loading prompt {prompt_name}: {str(e)}")


def extract_json(response_text: str) -> dict:
    try:
        # Remove any text before and after the JSON
        # Look for the first { and last }
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON structure found in response")

        json_str = response_text[start_idx:end_idx]

        # Clean the string if needed
        json_str = json_str.strip()

        # Parse JSON
        data = json.loads(json_str)

        return data

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        raise
    except ValueError as e:
        print(f"Validation error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
