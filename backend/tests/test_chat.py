import json
from collections.abc import Iterator
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import main
from services.model_service import ModelRefusalError


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
    assert response.headers["content-type"] == "application/x-ndjson"
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert [json.loads(line) for line in body.splitlines()] == [
        {"content": "An API "},
        {"content": "lets software communicate."},
    ]
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


def test_chat_logs_and_streams_error_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_stream() -> Iterator[str]:
        yield "Partial response"
        raise RuntimeError("Provider unavailable")

    service = Mock()
    service.create_chat_completion.return_value = failed_stream()
    logger = Mock()
    monkeypatch.setattr(main, "model_service", service)
    monkeypatch.setattr(main, "logger", logger)

    with TestClient(main.app) as client:
        response = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Question"}]},
        )

    assert response.status_code == 200
    assert [json.loads(line) for line in response.text.splitlines()] == [
        {"content": "Partial response"},
        {
            "error": {
                "message": main.CHAT_ERROR_MESSAGE,
                "retryable": True,
            }
        },
    ]
    logger.exception.assert_called_once_with(
        "Chat completion failed while streaming"
    )


def test_chat_streams_non_retryable_error_for_model_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refused_stream() -> Iterator[str]:
        raise ModelRefusalError("The model refused to answer the request")
        yield

    service = Mock()
    service.create_chat_completion.return_value = refused_stream()
    logger = Mock()
    monkeypatch.setattr(main, "model_service", service)
    monkeypatch.setattr(main, "logger", logger)

    with TestClient(main.app) as client:
        response = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Question"}]},
        )

    assert response.status_code == 200
    assert [json.loads(line) for line in response.text.splitlines()] == [
        {
            "error": {
                "message": main.CHAT_REFUSAL_MESSAGE,
                "retryable": False,
            }
        }
    ]
    logger.exception.assert_called_once_with(
        "Chat completion was refused while streaming"
    )


def test_chat_logs_and_returns_error_json_before_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = Mock()
    service.create_chat_completion.side_effect = RuntimeError(
        "Service unavailable"
    )
    logger = Mock()
    monkeypatch.setattr(main, "model_service", service)
    monkeypatch.setattr(main, "logger", logger)

    with TestClient(main.app) as client:
        response = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Question"}]},
        )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "message": main.CHAT_ERROR_MESSAGE,
            "retryable": True,
        }
    }
    logger.exception.assert_called_once_with(
        "Chat completion failed before streaming"
    )


def test_chat_returns_non_retryable_error_for_refusal_before_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = Mock()
    service.create_chat_completion.side_effect = ModelRefusalError(
        "The model refused to answer the request"
    )
    logger = Mock()
    monkeypatch.setattr(main, "model_service", service)
    monkeypatch.setattr(main, "logger", logger)

    with TestClient(main.app) as client:
        response = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Question"}]},
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": main.CHAT_REFUSAL_MESSAGE,
            "retryable": False,
        }
    }
    logger.exception.assert_called_once_with(
        "Chat completion was refused before streaming"
    )
