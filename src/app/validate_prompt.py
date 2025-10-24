def validate_prompt(prompt: str) -> str:
    if len(prompt) > 1000:
        raise ValueError("Prompt too long")
    return prompt.strip()