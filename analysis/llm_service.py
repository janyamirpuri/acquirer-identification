import logging
import os

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


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


def chat(prompt, client=None):
    if client is None:
        client = client_instance()

    logger.info("Invoking LLM with prompt length=%s", len(prompt))
    response = client.invoke(prompt)
    logger.info("LLM invocation completed successfully")
    return response.content
