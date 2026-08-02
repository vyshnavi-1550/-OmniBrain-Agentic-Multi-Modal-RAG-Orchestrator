"""
Vision-Language Model agent.

Real mode: sends the image (base64) to GPT-4o and asks it to describe /
extract numeric data from charts and tables.

Fallback mode (no OPENAI_API_KEY): returns a generic placeholder caption
derived from the image's page/position metadata so the rest of the
pipeline (indexing, routing, citation) is still fully exercised. Replace
with a local VLM (e.g. LLaVA via ollama) by swapping this function body.
"""
import base64
from config import USE_OPENAI, OPENAI_API_KEY


def describe_image(image_bytes: bytes, page_number: int, context_hint: str = "") -> str:
    if USE_OPENAI:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "This image was extracted from page "
                        f"{page_number} of a financial document. "
                        "Describe what it shows. If it is a chart or table, "
                        "extract the key numeric values precisely."
                    )},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64}"
                    }}
                ]
            }],
            max_tokens=500,
        )
        return resp.choices[0].message.content

    # --- Fallback: no VLM call made, placeholder description ---
    return (
        f"[VLM unavailable - placeholder] An image/chart extracted from page "
        f"{page_number}. {context_hint} Configure OPENAI_API_KEY to enable real "
        f"GPT-4o chart/table extraction."
    )


def answer_about_image(image_bytes: bytes, question: str) -> str:
    """Ask a specific question about one image (used by the Vision Agent at
    query time, not just ingestion time)."""
    if USE_OPENAI:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64}"
                    }}
                ]
            }],
            max_tokens=500,
        )
        return resp.choices[0].message.content
    return "[VLM unavailable] Set OPENAI_API_KEY to answer questions about this specific image."
