def validate_prompt(prompt: str) -> str:
    """Basic prompt validation"""
    if not prompt or len(prompt.strip()) == 0:
        raise ValueError("Prompt cannot be empty")
    if len(prompt) > 1000:
        raise ValueError("Prompt too long (max 1000 characters)")
    return prompt.strip()