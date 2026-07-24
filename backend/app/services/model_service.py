import logging
import os
from collections.abc import Iterator
from typing import Any

import httpx
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
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
logger = logging.getLogger("uvicorn.error")

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


def get_generated_chunks(
    content_blocks: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    generated_chunks: list[tuple[str, str]] = []
    for block in content_blocks:
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                generated_chunks.append(("text", text))
        elif block_type == "reasoning":
            reasoning = block.get("reasoning")
            if isinstance(reasoning, str):
                generated_chunks.append(("reasoning", reasoning))
    return generated_chunks


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
        returned_content = False

        try:
            stream = agent.stream(
                {"messages": agent_messages},
                config={"recursion_limit": 10},
                stream_mode="messages",
            )
            for token, metadata in stream:
                if not isinstance(token, AIMessageChunk):
                    continue

                for chunk_type, content in get_generated_chunks(
                    token.content_blocks
                ):
                    if not content:
                        continue

                    logger.info(
                        "Generated %s chunk: %r", chunk_type, content
                    )
                    returned_content = True
                    yield content
        except Exception as error:
            if is_server_connection_error(error):
                raise RuntimeError(
                    "The local model server is unavailable at "
                    f"{self.base_url}. Start llama-server and try again."
                ) from None
            raise

        if not returned_content:
            raise RuntimeError("The agent did not return an assistant response")


model_service = ModelService()
