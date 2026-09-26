
import os

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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


def build_chat_chain(system_prompt=None, client=None):
    """Create a minimal LCEL chat chain: prompt -> model -> string output."""
    if client is None:
        client = client_instance()

    prompt_messages = []
    if system_prompt is not None:
        prompt_messages.append(("system", system_prompt))
    prompt_messages.append(("human", "{user_input}"))

    prompt = ChatPromptTemplate.from_messages(prompt_messages)
    return prompt | client | StrOutputParser()


def build_tool_call_chain(client, tools=None):
    """Build an LCEL chain for tool-using models, preserving the existing agent flow."""
    model = client.bind_tools(tools) if tools else client
    return model


def chat(prompt, client=None):
    if client is None:
        client = client_instance()

    response = build_chat_chain(client=client).invoke({"user_input": prompt})
    return response
