from unittest.mock import Mock

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import services.model_service as model_service_module
from services.model_service import ModelService, convert_messages


def test_initialize_creates_model_and_agent_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_model = Mock()
    chat_openai = Mock(return_value=chat_model)
    agent = Mock()
    create_agent = Mock(return_value=agent)
    monkeypatch.setattr(model_service_module, "ChatOpenAI", chat_openai)
    monkeypatch.setattr(model_service_module, "create_agent", create_agent)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9090/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen-local")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("LLM_MAX_TOKENS", "1024")
    service = ModelService()

    first_result = service.initialize()
    second_result = service.initialize()

    assert first_result is chat_model
    assert second_result is chat_model
    chat_openai.assert_called_once_with(
        model="qwen-local",
        base_url="http://localhost:9090/v1",
        api_key="test-key",
        temperature=0.2,
        max_tokens=1024,
    )
    assert service.get_model() is chat_model
    assert service.get_agent() is agent
    assert [tool.name for tool in service.tools] == [
        "fighter_news",
        "fighter_stats",
        "fighter_wikipedia",
    ]
    create_agent.assert_called_once_with(
        model=chat_model,
        tools=service.tools,
        system_prompt=model_service_module.SYSTEM_PROMPT,
    )


def test_initialize_uses_local_server_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_openai = Mock(return_value=Mock())
    monkeypatch.setattr(model_service_module, "ChatOpenAI", chat_openai)
    monkeypatch.setattr(model_service_module, "create_agent", Mock())
    for variable in (
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_API_KEY",
        "LLM_TEMPERATURE",
        "LLM_MAX_TOKENS",
    ):
        monkeypatch.delenv(variable, raising=False)

    ModelService().initialize()

    chat_openai.assert_called_once_with(
        model="local-model",
        base_url="http://127.0.0.1:8080/v1",
        api_key="local-key",
        temperature=0.7,
        max_tokens=2048,
    )


def test_get_model_raises_when_model_is_not_initialized() -> None:
    service = ModelService()

    with pytest.raises(RuntimeError, match="The model has not been initialized"):
        service.get_model()


def test_convert_messages_preserves_supported_roles() -> None:
    result = convert_messages(
        [
            {"role": "system", "content": "System instructions"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Earlier answer"},
        ]
    )

    assert result == [
        SystemMessage(content="System instructions"),
        HumanMessage(content="Question"),
        AIMessage(content="Earlier answer"),
    ]


def test_convert_messages_rejects_unsupported_role() -> None:
    with pytest.raises(ValueError, match="Unsupported chat message role: tool"):
        convert_messages([{"role": "tool", "content": "result"}])


def test_create_chat_completion_returns_last_textual_assistant_message() -> None:
    agent = Mock()
    agent.invoke.return_value = {
        "messages": [
            HumanMessage(content="Question"),
            AIMessage(content="Earlier answer"),
            AIMessage(content=[{"type": "text", "text": "Final answer"}]),
        ]
    }
    service = ModelService()
    service.chat_model = Mock()
    service.agent = agent
    messages = [{"role": "user", "content": "Question"}]

    tokens = list(service.create_chat_completion(messages))

    assert tokens == ["Final answer"]
    invocation = agent.invoke.call_args
    assert invocation.args[0] == {
        "messages": [HumanMessage(content="Question")]
    }
    assert invocation.kwargs == {"config": {"recursion_limit": 10}}


def test_create_chat_completion_reports_unavailable_server() -> None:
    request = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions")
    agent = Mock()
    agent.invoke.side_effect = httpx.ConnectError(
        "Connection refused", request=request
    )
    service = ModelService()
    service.chat_model = Mock()
    service.agent = agent
    service.base_url = "http://127.0.0.1:8080/v1"

    with pytest.raises(
        RuntimeError,
        match=(
            "local model server is unavailable at "
            "http://127.0.0.1:8080/v1"
        ),
    ):
        list(
            service.create_chat_completion(
                [{"role": "user", "content": "Question"}]
            )
        )


def test_create_chat_completion_requires_final_assistant_message() -> None:
    agent = Mock()
    agent.invoke.return_value = {"messages": [HumanMessage(content="Question")]}
    service = ModelService()
    service.chat_model = Mock()
    service.agent = agent

    with pytest.raises(
        RuntimeError, match="agent did not return an assistant response"
    ):
        list(
            service.create_chat_completion(
                [{"role": "user", "content": "Question"}]
            )
        )
