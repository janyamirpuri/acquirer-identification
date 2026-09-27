import os

from langchain_openai import ChatOpenAI

def client_instance():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before using the LangChain client.")

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-5.2"),
        api_key=api_key,
        temperature=0,
    )


def chat(prompt, client=None):
    if client is None:
        client = client_instance()

    response = client.invoke(prompt)
    return response.content
