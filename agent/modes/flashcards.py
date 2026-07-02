import json
import re

from agent.llms import gemini
from agent.llm_router import estimate_cost_gbp

_MODEL = "gemini-1.5-flash"

_SYSTEM = """You create revision flashcards from content so someone can test themselves later.

Return ONLY a JSON array, no other text, no markdown code fences.
Each item must look like: {"question": "...", "answer": "..."}

Generate 6-10 cards covering the most important, testable points.
Questions should be short and specific. Answers should be short — one sentence or a few words."""


def _extract_cards(text: str) -> list:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    raw = match.group(0) if match else text
    return json.loads(raw)


def run(content: dict, gemini_key: str = None) -> dict:
    """Generate revision flashcards from content using Gemini Flash."""
    text = content["text"][:15000]
    prompt = f"{_SYSTEM}\n\nContent:\n\n{text}"

    result = gemini.complete(prompt, model=_MODEL, api_key=gemini_key)

    try:
        cards = _extract_cards(result["text"])
    except (json.JSONDecodeError, AttributeError):
        cards = []

    cost = estimate_cost_gbp(_MODEL, result["input_tokens"], result["output_tokens"])

    return {
        "cards": cards,
        "model": "Gemini Flash",
        "model_id": _MODEL,
        "cost_gbp": cost,
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
    }
