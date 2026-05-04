
import asyncio
import logging
from typing import AsyncGenerator
import google.generativeai as genai
from app.config.settings import GEMINI_API_KEY

logger = logging.getLogger(__name__)

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set.")

genai.configure(api_key=GEMINI_API_KEY)

_model = genai.GenerativeModel("gemini-1.5-flash")

MAX_CONTEXT_CHARS = 7_000


def _generate(prompt: str) -> str:
    try:
        response = _model.generate_content(prompt)
        return (response.text or "").strip()
    except Exception:
        logger.exception("LLM generation failed")
        return ""


async def generate_answer(prompt: str) -> str:
    """Async wrapper — generate_content is blocking, runs in thread."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_generate, prompt),
            timeout=30,
        )
    except asyncio.TimeoutError:
        logger.error("LLM generation timed out")
        return ""


async def stream_answer(prompt: str) -> AsyncGenerator[str, None]:
    """
    Stream answer tokens from Gemini.
    Yields text chunks as they arrive.
    generate_content(stream=True) runs in thread — does not block event loop.
    """
    def _stream():
        return _model.generate_content(prompt, stream=True)

    try:
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(_stream),
                timeout=30,
            )
        except asyncio.TimeoutError:
            logger.error("LLM streaming init timed out")
            return
        for chunk in response:
            text = (chunk.text or "")
            if text:
                yield text
    except Exception:
        logger.exception("LLM streaming failed")
        yield ""