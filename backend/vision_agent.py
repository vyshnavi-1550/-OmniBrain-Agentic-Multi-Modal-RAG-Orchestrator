"""
Vision-Language Model agent.

Priority order:
  1. GPT-4o (if OPENAI_API_KEY is set) -- highest quality, paid.
  2. Gemini 1.5 Flash (if GOOGLE_API_KEY is set) -- free-tier vision model,
     used as the default real VLM for this project since it requires no
     paid API key.
  3. Fallback placeholder -- if neither key is configured, returns a
     generic caption derived from page/position metadata so the rest of
     the pipeline (indexing, routing, citation) is still fully exercised.
"""
import base64
from config import USE_OPENAI, OPENAI_API_KEY, USE_GEMINI, GOOGLE_API_KEY


def describe_image(image_bytes: bytes, page_number: int, context_hint: str = "") -> str:
    prompt = (
        "This image was extracted from page "
        f"{page_number} of a financial document. "
        "Describe what it shows. If it is a chart or table, "
        "extract the key numeric values precisely."
    )

    if USE_OPENAI:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64}"
                    }}
                ]
            }],
            max_tokens=500,
        )
        return resp.choices[0].message.content

    if USE_GEMINI:
        return _describe_image_gemini(image_bytes, prompt)

    # --- Fallback: no VLM call made, placeholder description ---
    return (
        f"[VLM unavailable - placeholder] An image/chart extracted from page "
        f"{page_number}. {context_hint} Configure OPENAI_API_KEY or GOOGLE_API_KEY "
        f"to enable real chart/table extraction."
    )


def _describe_image_gemini(image_bytes: bytes, prompt: str) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-3.5-flash-lite")
        image_part = {"mime_type": "image/png", "data": image_bytes}
        resp = model.generate_content([prompt, image_part])
        return resp.text
    except Exception as e:
        return f"[Gemini VLM error: {e}] Falling back -- could not extract chart/table data."


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

    if USE_GEMINI:
        return _describe_image_gemini(image_bytes, question)

    return "[VLM unavailable] Set OPENAI_API_KEY or GOOGLE_API_KEY to answer questions about this specific image."