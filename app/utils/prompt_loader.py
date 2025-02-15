import os
from pathlib import Path


class PromptLoader:
    @staticmethod
    def load_prompt(prompt_name: str) -> str:
        """
        Load prompt from a text file.

        Args:
            prompt_name: Name of the prompt file without .txt extension

        Returns:
            str: Content of the prompt file
        """
        try:
            # Get the absolute path to the prompts directory
            prompts_dir = Path(__file__).parent.parent / 'config' / 'prompts'
            prompt_path = prompts_dir / f"{prompt_name}.txt"

            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

            with open(prompt_path, 'r', encoding='utf-8') as file:
                return file.read().strip()

        except Exception as e:
            raise Exception(f"Error loading prompt {prompt_name}: {str(e)}")