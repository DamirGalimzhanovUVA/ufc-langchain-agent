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


def test_create_chat_completion_returns_assistant_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Mock()
    model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "An API lets software communicate."}}]
    }
    monkeypatch.setattr(model_service_module, "Llama", Mock(return_value=model))
    service = ModelService()
    service.initialize("model.gguf")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is an API?"},
    ]

    response = service.create_chat_completion(messages)

    assert response == "An API lets software communicate."
    model.create_chat_completion.assert_called_once_with(messages=messages)


def test_create_chat_completion_raises_for_empty_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Mock()
    model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": None}}]
    }
    monkeypatch.setattr(model_service_module, "Llama", Mock(return_value=model))
    service = ModelService()
    service.initialize("model.gguf")

    with pytest.raises(RuntimeError, match="empty assistant response"):
        service.create_chat_completion([{"role": "user", "content": "Hello"}])
