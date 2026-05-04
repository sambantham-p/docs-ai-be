import asyncio
import logging
from typing import AsyncGenerator
import google.generativeai as genai
from app.config.settings import GEMINI_API_KEY, LLM_TIMEOUT
import time


logger = logging.getLogger(__name__)
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set.")
genai.configure(api_key=GEMINI_API_KEY)
_model = genai.GenerativeModel("gemini-1.5-flash")


def _generate(prompt: str) -> str:
    try:
        response = _model.generate_content(prompt)
        return (response.text or "").strip()
    except Exception:
        logger.exception("LLM generation failed")
        return ""


# Async wrapper, Runs blocking Gemini API call in a thread and enforces a timeout.
async def generate_answer(prompt: str) -> str:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_generate, prompt),
            timeout=LLM_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error("LLM generation timed out")
        return ""


# Streams response chunks from Gemini without blocking the event loop.
async def stream_answer(prompt: str) -> AsyncGenerator[str, None]:
    def _stream():
        return _model.generate_content(prompt, stream=True)
    try:
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(_stream),
                timeout=LLM_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error("LLM streaming init timed out")
            yield "[ERROR]"
            return
        start = time.time()
        for chunk in response:
            if time.time() - start > 60:
                logger.error("LLM streaming timed out during iteration")
                break
            text = (chunk.text or "")
            if text:
                yield text
    except Exception:
        logger.exception("LLM streaming failed")
        yield "[ERROR]"
        return