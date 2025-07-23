"""Factory for lightweight Claude‑3‑Haiku chat call."""
import os
import anthropic

def get_lightweight_chat_call():
    """Return a lightweight chat call function using Claude-3-Haiku."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    def _call(prompt: str) -> str:
        resp = client.messages.create(
            model="claude-3-haiku-20240307",
            temperature=0.7,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    return _call

__all__ = ["get_lightweight_chat_call"]
