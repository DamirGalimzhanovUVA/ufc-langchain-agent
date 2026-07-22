from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import main


def test_chat_returns_model_response(monkeypatch: pytest.MonkeyPatch) -> None:
    service = Mock()
    service.create_chat_completion.return_value = "An API lets software communicate."
    monkeypatch.setenv("LLAMA_MODEL_PATH", "model.gguf")
    monkeypatch.setattr(main, "model_service", service)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is an API?"},
    ]

    with TestClient(main.app) as client:
        response = client.post("/chat", json={"messages": messages})

    assert response.status_code == 200
    assert response.json() == {"response": "An API lets software communicate."}
    service.initialize.assert_called_once_with("model.gguf")
    service.create_chat_completion.assert_called_once_with(messages)


def test_chat_rejects_invalid_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    service = Mock()
    monkeypatch.setenv("LLAMA_MODEL_PATH", "model.gguf")
    monkeypatch.setattr(main, "model_service", service)

    with TestClient(main.app) as client:
        response = client.post(
            "/chat", json={"messages": [{"role": "user"}]}
        )

    assert response.status_code == 422
    service.create_chat_completion.assert_not_called()
