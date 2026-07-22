from unittest.mock import Mock

import pytest

import services.model_service as model_service_module
from services.model_service import ModelService


def test_initialize_creates_model_once(monkeypatch: pytest.MonkeyPatch) -> None:
    model = Mock()
    llama = Mock(return_value=model)
    monkeypatch.setattr(model_service_module, "Llama", llama)
    service = ModelService()

    first_result = service.initialize("first-model.gguf")
    second_result = service.initialize("second-model.gguf")

    assert first_result is model
    assert second_result is model
    llama.assert_called_once_with(model_path="first-model.gguf")


def test_get_model_returns_initialized_model(monkeypatch: pytest.MonkeyPatch) -> None:
    model = Mock()
    monkeypatch.setattr(model_service_module, "Llama", Mock(return_value=model))
    service = ModelService()

    service.initialize("model.gguf")

    assert service.get_model() is model


def test_get_model_raises_when_model_is_not_initialized() -> None:
    service = ModelService()

    with pytest.raises(RuntimeError, match="The model has not been initialized"):
        service.get_model()
