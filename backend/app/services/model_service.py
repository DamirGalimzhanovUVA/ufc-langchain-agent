import json
import os
from collections.abc import Iterator
from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from llama_cpp import Llama

from services.fighter_stats_provider import HttpFighterStatsProvider
from services.fighter_tools import (
    TavilyNewsClient,
    WikipediaClient,
    create_fighter_tools,
)

ChatMessage = dict[str, str]


class LlamaCppChatModel(BaseChatModel):
    model: Any
    max_tokens: int = 2048

    @property
    def _llm_type(self) -> str:
        return "existing-llama-cpp"

    def bind_tools(
        self,
        tools: list[dict[str, Any] | type | Any | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Any:
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=formatted_tools, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return generate_from_stream(
            self._stream(messages, stop, run_manager, **kwargs)
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts = cast(
            list[dict[str, Any]], convert_to_openai_messages(messages)
        )
        completion_kwargs = {"max_tokens": self.max_tokens, **kwargs}
        completion = self.model.create_chat_completion(
            messages=message_dicts,
            stream=True,
            stop=stop,
            **completion_kwargs,
        )

        for raw_chunk in cast(Iterator[dict[str, Any]], completion):
            choices = raw_chunk.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            tool_call_chunks = [
                {
                    "name": tool_call.get("function", {}).get("name"),
                    "args": tool_call.get("function", {}).get("arguments"),
                    "id": tool_call.get("id"),
                    "index": tool_call.get("index"),
                }
                for tool_call in delta.get("tool_calls", [])
            ]
            if (
                (not isinstance(content, str) or not content)
                and not tool_call_chunks
            ):
                continue

            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=content if isinstance(content, str) else "",
                    tool_call_chunks=tool_call_chunks,
                )
            )
            if run_manager is not None and isinstance(content, str) and content:
                run_manager.on_llm_new_token(content, chunk=chunk)
            yield chunk


class ModelService:
    def __init__(self) -> None:
        self.chat_model: BaseChatModel | None = None
        self.tools: list[BaseTool] = []

    def initialize(self, model_path: str) -> BaseChatModel:
        if self.chat_model is None:
            llama_model = Llama(
                model_path=model_path,
                n_ctx=8192,
                n_gpu_layers=-1,
                verbose=True,
            )
            self.chat_model = LlamaCppChatModel(model=llama_model)
            stats_provider = HttpFighterStatsProvider(
                api_url=os.environ.get("FIGHTER_STATS_API_URL", ""),
                api_key=os.environ.get("FIGHTER_STATS_API_KEY"),
            )
            self.tools = create_fighter_tools(
                news_client=TavilyNewsClient(os.environ.get("TAVILY_API_KEY")),
                stats_provider=stats_provider,
                wikipedia_client=WikipediaClient(),
            )

        return self.chat_model

    def get_model(self) -> BaseChatModel:
        if self.chat_model is None:
            raise RuntimeError("The model has not been initialized")

        return self.chat_model

    def create_chat_completion(self, messages: list[ChatMessage]) -> Iterator[str]:
        model = self.get_model()
        tool_model = model.bind_tools(self.tools) if self.tools else model
        tools_by_name = {tool.name: tool for tool in self.tools}
        conversation: list[Any] = list(messages)
        tool_rounds_remaining = 5

        while tool_rounds_remaining > 0:
            tool_rounds_remaining -= 1
            chunks = list(tool_model.stream(list(conversation)))
            if not chunks:
                return

            response = cast(AIMessage, chunks[0])
            for chunk in chunks[1:]:
                response = response + chunk

            if not response.tool_calls:
                for chunk in chunks:
                    content = chunk.content
                    if isinstance(content, str) and content:
                        yield content
                return

            conversation.append(response)
            for tool_call in response.tool_calls:
                tool = tools_by_name.get(tool_call["name"])
                if tool is None:
                    result: Any = {
                        "error": f"Unknown tool: {tool_call['name']}"
                    }
                else:
                    try:
                        result = tool.invoke(tool_call["args"])
                    except Exception as error:
                        result = {"error": str(error)}

                conversation.append(
                    ToolMessage(
                        content=json.dumps(result),
                        tool_call_id=tool_call["id"],
                    )
                )

        yield "I could not complete the request after several tool calls."


model_service = ModelService()
