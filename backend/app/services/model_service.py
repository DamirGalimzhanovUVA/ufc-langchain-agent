from llama_cpp import Llama

ChatMessage = dict[str, str]


class ModelService:
    def __init__(self) -> None:
        self.model: Llama | None = None

    def initialize(self, model_path: str) -> Llama:
        if self.model is None:
            self.model = Llama(model_path=model_path)

        return self.model

    def get_model(self) -> Llama:
        if self.model is None:
            raise RuntimeError("The model has not been initialized")

        return self.model

    def create_chat_completion(self, messages: list[ChatMessage]) -> str:
        model = self.get_model()
        completion = model.create_chat_completion(messages=messages)
        content = completion["choices"][0]["message"]["content"]

        if content is None:
            raise RuntimeError("The model returned an empty assistant response")

        return content


model_service = ModelService()
