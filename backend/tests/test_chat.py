from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import main


def test_chat_streams_model_response(monkeypatch: pytest.MonkeyPatch) -> None:
    service = Mock()
    service.create_chat_completion.return_value = iter(
        ["An API ", "lets software communicate."]
    )
    monkeypatch.setattr(main, "model_service", service)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is an API?"},
    ]

    with TestClient(main.app) as client:
        with client.stream("POST", "/chat", json={"messages": messages}) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert body == "An API lets software communicate."
    service.initialize.assert_called_once_with()
    service.create_chat_completion.assert_called_once_with(messages)


def test_chat_rejects_invalid_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    service = Mock()
    monkeypatch.setattr(main, "model_service", service)

    with TestClient(main.app) as client:
        response = client.post(
            "/chat", json={"messages": [{"role": "user"}]}
        )

    assert response.status_code == 422
    service.create_chat_completion.assert_not_called()
