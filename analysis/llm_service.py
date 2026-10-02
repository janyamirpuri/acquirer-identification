import logging
import os
import time

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)
RETRYABLE_EXCEPTIONS = (TimeoutError, ConnectionError, OSError)


def client_instance():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before using the LangChain client.")

    logger.info("Initializing ChatOpenAI model=%s with temperature=0, seed=42",
                os.getenv("OPENAI_MODEL", "gpt-5.2"))

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-5.2"),
        api_key=api_key,
        temperature=0,
        seed=42,
    )


def chat(prompt, client=None, max_retries: int = 3, retry_delay: float = 1.0):
    if client is None:
        client = client_instance()

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            logger.info("Invoking LLM with prompt length=%s (attempt %s/%s)", len(prompt), attempt + 1, max_retries + 1)
            response = client.invoke(prompt)
            logger.info("LLM invocation completed successfully")
            return response.content
        except RETRYABLE_EXCEPTIONS as exc:
            last_error = exc
            if attempt >= max_retries:
                logger.exception("LLM invocation failed after %s attempt(s)", max_retries + 1)
                raise
            delay = retry_delay * (2 ** attempt)
            logger.warning("LLM invocation failed with %s; retrying in %ss", type(exc).__name__, delay)
            time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError("LLM invocation failed without an error")
