import logging
import os
from collections.abc import Iterator
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, after_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from services.fighter_stats_provider import HttpFighterStatsProvider
from services.fighter_tools import (
    TavilyNewsClient,
    WikipediaClient,
    create_fighter_tools,
)

ChatMessage = dict[str, str]
logger = logging.getLogger("uvicorn.error")


class ModelResponseError(RuntimeError):
    def __init__(
        self,
        message: str,
        model_response: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.model_response = model_response


class ModelRefusalError(ModelResponseError):
    pass


class EmptyModelResponseError(ModelResponseError):
    pass


SYSTEM_PROMPT = """You are a UFC research assistant.
Use the Wikipedia tool for fighter background and career information.
Use the fighter stats tool for structured fighter statistics.
Use the fight description tool when asked for the background, stakes, or
description of a specific matchup covered by an MMA Fighting live blog.
Call the fight description tool no more than once per user request.
Use the Tavily news tool for recent developments. When using it, generate a
focused, standalone search query that preserves every person and topic from the
user's question. After each search, determine whether the result content answers
the user's specific question. If it does not, refine the query based on what is
missing and search again. Make no more than two retries after the original search,
for a maximum of three Tavily searches per user request. Do not retry after finding
a relevant answer. If all three searches fail to find one, clearly say that the
answer could not be found.
Do not call tools when the answer can be produced from the existing conversation.
If a tool fails, explain the failure instead of inventing results.
After using tools, produce a normal user-facing answer."""


@after_model
def log_news_search_decision(
    state: dict[str, Any], runtime: Any
) -> None:
    messages = state.get("messages", [])
    if not messages or not isinstance(messages[-1], AIMessage):
        return

    received_news_results = False
    for message in reversed(messages[:-1]):
        if not isinstance(message, ToolMessage):
            break
        if message.name == "fighter_news":
            received_news_results = True

    if not received_news_results:
        return

    model_message = messages[-1]
    retry_queries = [
        tool_call.get("args", {}).get("query")
        for tool_call in model_message.tool_calls
        if tool_call.get("name") == "fighter_news"
    ]
    retry_queries = [
        query for query in retry_queries if isinstance(query, str)
    ]
    action = "retry" if retry_queries else "stop"
    logger.info(
        "Model news search decision: action=%s next_queries=%s response=%r",
        action,
        retry_queries,
        model_message.content,
    )


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


def get_refusal(
    content_blocks: list[dict[str, Any]],
    additional_kwargs: dict[str, Any],
) -> str | None:
    refusal = additional_kwargs.get("refusal")
    if isinstance(refusal, str):
        return refusal

    for block in content_blocks:
        if block.get("type") == "refusal":
            refusal = block.get("refusal")
        elif block.get("type") == "non_standard":
            value = block.get("value")
            if not isinstance(value, dict) or value.get("type") != "refusal":
                continue
            refusal = value.get("refusal")
        else:
            continue

        if isinstance(refusal, str):
            return refusal

    return None


class ModelService:
    def __init__(self) -> None:
        self.chat_model: BaseChatModel | None = None
        self.agent: Any = None
        self.tools: list[BaseTool] = []

    def initialize(self) -> BaseChatModel:
        if self.chat_model is None:
            self.chat_model = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "gpt-5-nano"),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            )
            stats_provider = HttpFighterStatsProvider(
                api_key=os.environ.get("FIGHTER_STATS_API_KEY"),
            )
            tavily_api_key = os.environ.get("TAVILY_API_KEY")
            self.tools = create_fighter_tools(
                news_client=TavilyNewsClient(tavily_api_key),
                stats_provider=stats_provider,
                wikipedia_client=WikipediaClient(),
                tavily_api_key=tavily_api_key,
            )
            self.agent = create_agent(
                model=self.chat_model,
                tools=self.tools,
                system_prompt=SYSTEM_PROMPT,
                middleware=[
                    log_news_search_decision,
                    ToolCallLimitMiddleware(
                        tool_name="fighter_news",
                        run_limit=3,
                        exit_behavior="continue",
                    ),
                    ToolCallLimitMiddleware(
                        tool_name="fight_description",
                        run_limit=1,
                        exit_behavior="continue",
                    ),
                ],
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
        graph_steps: set[int] = set()
        graph_tasks = 0
        model_calls = 0
        tool_calls = 0
        last_model_response: dict[str, Any] | None = None

        stream = agent.stream(
            {"messages": agent_messages},
            config={"recursion_limit": 30},
            stream_mode=["messages", "debug"],
        )
        try:
            for stream_mode, event in stream:
                if stream_mode == "debug":
                    if event.get("type") != "task":
                        continue

                    step = event.get("step")
                    node = event.get("payload", {}).get("name")
                    if isinstance(step, int):
                        graph_steps.add(step)
                    graph_tasks += 1
                    if node == "model":
                        model_calls += 1
                    elif node == "tools":
                        tool_calls += 1
                    logger.info(
                        "Graph hop: step=%s node=%s",
                        step,
                        node,
                    )
                    continue

                token, metadata = event
                if not isinstance(token, AIMessageChunk):
                    continue

                last_model_response = token.model_dump(mode="json")
                refusal = get_refusal(
                    token.content_blocks,
                    token.additional_kwargs,
                )
                if refusal is not None:
                    logger.info("Generated refusal: %r", refusal)
                    raise ModelRefusalError(
                        "The model refused to answer the request",
                        last_model_response,
                    )

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
        finally:
            logger.info(
                "Graph execution summary: supersteps=%s graph_tasks=%s "
                "model_calls=%s tool_calls=%s",
                len(graph_steps),
                graph_tasks,
                model_calls,
                tool_calls,
            )

        if not returned_content:
            raise EmptyModelResponseError(
                "The agent did not return an assistant response",
                last_model_response,
            )


model_service = ModelService()
