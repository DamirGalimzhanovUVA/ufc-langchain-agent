import os
from collections.abc import Iterator
from typing import Any

import httpx
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from openai import APIConnectionError

from services.fighter_stats_provider import HttpFighterStatsProvider
from services.fighter_tools import (
    TavilyNewsClient,
    WikipediaClient,
    create_fighter_tools,
)

ChatMessage = dict[str, str]

SYSTEM_PROMPT = """You are a UFC research assistant.
Use the Wikipedia tool for fighter background and career information.
Use the fighter stats tool for structured fighter statistics.
Use the Tavily news tool for recent developments.
Do not call tools when the answer can be produced from the existing conversation.
If a tool fails, explain the failure instead of inventing results.
After using tools, produce a normal user-facing answer."""


def convert_messages(messages: list[ChatMessage]) -> list[BaseMessage]:
    converted_messages: list[BaseMessage] = []
    message_types = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in message_types:
            raise ValueError(f"Unsupported chat message role: {role}")
        if not isinstance(content, str):
            raise ValueError("Chat message content must be a string")
        converted_messages.append(message_types[role](content=content))

    return converted_messages


def get_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)
    return "".join(text_parts)


def is_server_connection_error(error: BaseException) -> bool:
    current_error: BaseException | None = error
    while current_error is not None:
        if isinstance(
            current_error,
            (APIConnectionError, httpx.ConnectError, httpx.TimeoutException),
        ):
            return True
        current_error = current_error.__cause__ or current_error.__context__
    return False


class ModelService:
    def __init__(self) -> None:
        self.chat_model: BaseChatModel | None = None
        self.agent: Any = None
        self.tools: list[BaseTool] = []
        self.base_url = ""

    def initialize(self) -> BaseChatModel:
        if self.chat_model is None:
            self.base_url = os.getenv(
                "LLM_BASE_URL", "http://127.0.0.1:8080/v1"
            )
            self.chat_model = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "local-model"),
                base_url=self.base_url,
                api_key=os.getenv("LLM_API_KEY", "local-key"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            )
            stats_provider = HttpFighterStatsProvider(
                api_key=os.environ.get("FIGHTER_STATS_API_KEY"),
            )
            self.tools = create_fighter_tools(
                news_client=TavilyNewsClient(os.environ.get("TAVILY_API_KEY")),
                stats_provider=stats_provider,
                wikipedia_client=WikipediaClient(),
            )
            self.agent = create_agent(
                model=self.chat_model,
                tools=self.tools,
                system_prompt=SYSTEM_PROMPT,
            )

        return self.chat_model

    def get_model(self) -> BaseChatModel:
        if self.chat_model is None:
            raise RuntimeError("The model has not been initialized")

        return self.chat_model

    def get_agent(self) -> Any:
        self.get_model()
        if self.agent is None:
            raise RuntimeError("The agent has not been initialized")
        return self.agent

    def create_chat_completion(
        self, messages: list[ChatMessage]
    ) -> Iterator[str]:
        agent = self.get_agent()
        agent_messages = convert_messages(messages)

        try:
            result = agent.invoke(
                {"messages": agent_messages},
                config={"recursion_limit": 10},
            )
        except Exception as error:
            if is_server_connection_error(error):
                raise RuntimeError(
                    "The local model server is unavailable at "
                    f"{self.base_url}. Start llama-server and try again."
                ) from None
            raise

        result_messages = result.get("messages", [])
        for message in reversed(result_messages):
            if not isinstance(message, AIMessage):
                continue
            content = get_text_content(message.content)
            if content:
                yield content
                return

        raise RuntimeError("The agent did not return an assistant response")


model_service = ModelService()
