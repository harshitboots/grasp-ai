from agent.llms import claude
from agent.llm_router import estimate_cost_gbp

_MODEL = "claude-haiku-4-5-20251001"

_SYSTEM_TEMPLATE = """You are answering questions about a specific piece of content the user uploaded to Grasp.

Only answer using the content below. If the answer isn't in the content, say so clearly — don't make things up.

Keep answers short and direct. Reference specific parts of the content when useful.

--- CONTENT START ---
{content}
--- CONTENT END ---"""


def run(content: dict, question: str, history: list = None, anthropic_key: str = None) -> dict:
    """Answer a question about the content, using prior chat turns for context."""
    history = history or []
    text = content["text"][:15000]

    system = _SYSTEM_TEMPLATE.format(content=text)
    messages = history + [{"role": "user", "content": question}]

    result = claude.chat(
        system=system,
        messages=messages,
        model=_MODEL,
        max_tokens=600,
        api_key=anthropic_key,
    )

    cost = estimate_cost_gbp(_MODEL, result["input_tokens"], result["output_tokens"])

    return {
        "output": result["text"],
        "model": "Claude Haiku",
        "model_id": _MODEL,
        "cost_gbp": cost,
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
    }
