from collections.abc import Iterator
from typing import Any, cast

from llama_cpp import Llama

ChatMessage = dict[str, str]


class ModelService:
    def __init__(self) -> None:
        self.model: Llama | None = None

    def initialize(self, model_path: str) -> Llama:
        if self.model is None:
            self.model = Llama(
                model_path=model_path,
                n_ctx=8192,
                n_gpu_layers=-1,
                verbose=True,
            )
        

        return self.model

    def get_model(self) -> Llama:
        if self.model is None:
            raise RuntimeError("The model has not been initialized")

        return self.model

    def create_chat_completion(self, messages: list[ChatMessage]) -> Iterator[str]:
        model = self.get_model()
        completion = model.create_chat_completion(
            messages=messages, 
            stream=True, 
            max_tokens=2048
        )

        for chunk in cast(Iterator[dict[str, Any]], completion):
            content = chunk["choices"][0]["delta"].get("content")

            if content:
                yield content


model_service = ModelService()
