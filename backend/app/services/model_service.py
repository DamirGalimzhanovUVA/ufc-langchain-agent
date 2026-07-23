from collections.abc import Iterator
from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    convert_to_openai_messages,
)
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from llama_cpp import Llama

ChatMessage = dict[str, str]


class LlamaCppChatModel(BaseChatModel):
    model: Any
    max_tokens: int = 2048

    @property
    def _llm_type(self) -> str:
        return "existing-llama-cpp"

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
            if not isinstance(content, str) or not content:
                continue

            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=content)
            )
            if run_manager is not None:
                run_manager.on_llm_new_token(content, chunk=chunk)
            yield chunk


class ModelService:
    def __init__(self) -> None:
        self.chat_model: BaseChatModel | None = None

    def initialize(self, model_path: str) -> BaseChatModel:
        if self.chat_model is None:
            llama_model = Llama(
                model_path=model_path,
                n_ctx=8192,
                n_gpu_layers=-1,
                verbose=True,
            )
            self.chat_model = LlamaCppChatModel(model=llama_model)

        return self.chat_model

    def get_model(self) -> BaseChatModel:
        if self.chat_model is None:
            raise RuntimeError("The model has not been initialized")

        return self.chat_model

    def create_chat_completion(self, messages: list[ChatMessage]) -> Iterator[str]:
        model = self.get_model()
        for chunk in model.stream(messages):
            content = chunk.content
            if isinstance(content, str) and content:
                yield content


model_service = ModelService()
