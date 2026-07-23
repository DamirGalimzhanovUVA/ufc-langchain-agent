from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage

import services.model_service as model_service_module
from services.model_service import LlamaCppChatModel, ModelService


def test_initialize_creates_model_once(monkeypatch: pytest.MonkeyPatch) -> None:
    model = Mock()
    llama = Mock(return_value=model)
    monkeypatch.setattr(model_service_module, "Llama", llama)
    service = ModelService()

    first_result = service.initialize("first-model.gguf")
    second_result = service.initialize("second-model.gguf")

    assert first_result is model
    assert second_result is model
    llama.assert_called_once_with(
        model_path="first-model.gguf",
        n_ctx=8192,
        n_gpu_layers=-1,
        verbose=True,
    )
    assert service.get_chat_model().model is model


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


def test_get_chat_model_raises_when_model_is_not_initialized() -> None:
    service = ModelService()

    with pytest.raises(
        RuntimeError, match="The chat model has not been initialized"
    ):
        service.get_chat_model()


def test_llama_cpp_chat_model_streams_assistant_response() -> None:
    model = Mock()
    model.create_chat_completion.return_value = iter(
        [
            {"choices": [{"delta": {"content": "An API "}}]},
            {"choices": [{"delta": {"content": "lets software communicate."}}]},
            {"choices": [{"delta": {}}]},
        ]
    )
    messages = [
        SystemMessage("You are a helpful assistant."),
        HumanMessage("What is an API?"),
    ]
    chat_model = LlamaCppChatModel(model=model)

    tokens = [chunk.content for chunk in chat_model.stream(messages)]

    assert tokens == ["An API ", "lets software communicate.", ""]
    model.create_chat_completion.assert_called_once_with(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is an API?"},
        ],
        stream=True,
        max_tokens=2048,
        stop=None,
    )


def test_create_chat_completion_uses_langchain_chat_model() -> None:
    chat_model = Mock()
    chat_model.stream.return_value = iter(
        [
            AIMessageChunk(content="An API "),
            AIMessageChunk(content="lets software communicate."),
        ]
    )
    service = ModelService()
    service.chat_model = chat_model
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is an API?"},
    ]

    tokens = list(service.create_chat_completion(messages))

    assert tokens == ["An API ", "lets software communicate."]
    chat_model.stream.assert_called_once_with(messages)
