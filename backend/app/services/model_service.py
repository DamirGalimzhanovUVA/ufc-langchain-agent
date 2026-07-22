from llama_cpp import Llama


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


model_service = ModelService()
