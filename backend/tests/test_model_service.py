from collections.abc import Iterator
from unittest.mock import Mock, call

import httpx
import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

import services.model_service as model_service_module
from services.model_service import ModelService, convert_messages


def test_initialize_creates_model_and_agent_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_model = Mock()
    chat_openai = Mock(return_value=chat_model)
    agent = Mock()
    create_agent = Mock(return_value=agent)
    tool_call_limit = Mock(return_value=Mock())
    monkeypatch.setattr(model_service_module, "ChatOpenAI", chat_openai)
    monkeypatch.setattr(model_service_module, "create_agent", create_agent)
    monkeypatch.setattr(
        model_service_module, "ToolCallLimitMiddleware", tool_call_limit
    )
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
        "fight_description",
    ]
    tool_call_limit.assert_called_once_with(
        tool_name="fighter_news",
        run_limit=3,
        exit_behavior="continue",
    )
    create_agent.assert_called_once_with(
        model=chat_model,
        tools=service.tools,
        system_prompt=model_service_module.SYSTEM_PROMPT,
        middleware=[
            model_service_module.log_news_search_decision,
            tool_call_limit.return_value,
        ],
    )


def test_system_prompt_requires_up_to_two_relevant_news_retries() -> None:
    assert "determine whether the result content answers" in (
        model_service_module.SYSTEM_PROMPT
    )
    assert "no more than two retries after the original search" in (
        model_service_module.SYSTEM_PROMPT
    )
    assert "maximum of three Tavily searches per user request" in (
        model_service_module.SYSTEM_PROMPT
    )


def test_news_search_decision_logs_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = Mock()
    monkeypatch.setattr(model_service_module, "logger", logger)
    state = {
        "messages": [
            HumanMessage(content="What did Makhachev say about Garry?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fighter_news",
                        "args": {"query": "first query"},
                        "id": "call-1",
                    }
                ],
            ),
            ToolMessage(
                content='{"results": []}',
                name="fighter_news",
                tool_call_id="call-1",
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fighter_news",
                        "args": {"query": "refined query"},
                        "id": "call-2",
                    }
                ],
            ),
        ]
    }

    model_service_module.log_news_search_decision.after_model(state, Mock())

    logger.info.assert_called_once_with(
        "Model news search decision: action=%s next_queries=%s response=%r",
        "retry",
        ["refined query"],
        "",
    )


def test_news_search_decision_logs_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = Mock()
    monkeypatch.setattr(model_service_module, "logger", logger)
    response = "No direct comment was found."
    state = {
        "messages": [
            ToolMessage(
                content='{"results": []}',
                name="fighter_news",
                tool_call_id="call-1",
            ),
            AIMessage(content=response),
        ]
    }

    model_service_module.log_news_search_decision.after_model(state, Mock())

    logger.info.assert_called_once_with(
        "Model news search decision: action=%s next_queries=%s response=%r",
        "stop",
        [],
        response,
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


def test_create_chat_completion_streams_and_logs_chunks_as_they_arrive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream_finished = False

    def stream(
        *args: object, **kwargs: object
    ) -> Iterator[tuple[AIMessageChunk, dict[str, str]]]:
        nonlocal stream_finished
        yield (
            AIMessageChunk(
                content="",
                additional_kwargs={"reasoning_content": "Thinking "},
            ),
            {"langgraph_node": "model"},
        )
        yield (
            AIMessageChunk(
                content=[{"type": "reasoning", "reasoning": "carefully. "}]
            ),
            {"langgraph_node": "model"},
        )
        yield AIMessageChunk(content="Final "), {"langgraph_node": "model"}
        yield AIMessageChunk(content="answer"), {"langgraph_node": "model"}
        stream_finished = True

    agent = Mock()
    agent.stream.side_effect = stream
    logger = Mock()
    monkeypatch.setattr(model_service_module, "logger", logger)
    service = ModelService()
    service.chat_model = Mock()
    service.agent = agent
    messages = [{"role": "user", "content": "Question"}]

    tokens = service.create_chat_completion(messages)

    assert next(tokens) == "Thinking "
    assert stream_finished is False
    logger.info.assert_called_once_with(
        "Generated %s chunk: %r", "reasoning", "Thinking "
    )

    assert list(tokens) == ["carefully. ", "Final ", "answer"]
    assert stream_finished is True
    assert logger.info.call_args_list == [
        call("Generated %s chunk: %r", "reasoning", "Thinking "),
        call("Generated %s chunk: %r", "reasoning", "carefully. "),
        call("Generated %s chunk: %r", "text", "Final "),
        call("Generated %s chunk: %r", "text", "answer"),
    ]
    invocation = agent.stream.call_args
    assert invocation.args[0] == {
        "messages": [HumanMessage(content="Question")]
    }
    assert invocation.kwargs == {
        "config": {"recursion_limit": 10},
        "stream_mode": "messages",
    }


def test_create_chat_completion_ignores_non_text_agent_events() -> None:
    agent = Mock()
    agent.stream.return_value = iter(
        [
            (
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": "fighter_stats",
                            "args": '{"fighter_name": "José Aldo"}',
                            "id": "call-1",
                            "index": 0,
                        }
                    ],
                ),
                {"langgraph_node": "model"},
            ),
            (
                ToolMessage(content='{"wins": 32}', tool_call_id="call-1"),
                {"langgraph_node": "tools"},
            ),
            (
                AIMessageChunk(
                    content=[{"type": "text", "text": "José Aldo has 32 wins."}]
                ),
                {"langgraph_node": "model"},
            ),
        ]
    )
    service = ModelService()
    service.chat_model = Mock()
    service.agent = agent

    tokens = list(
        service.create_chat_completion(
            [{"role": "user", "content": "How many wins does José Aldo have?"}]
        )
    )

    assert tokens == ["José Aldo has 32 wins."]


def test_create_chat_completion_reports_unavailable_server() -> None:
    request = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions")
    agent = Mock()
    agent.stream.side_effect = httpx.ConnectError(
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
    agent.stream.return_value = iter(
        [(HumanMessage(content="Question"), {"langgraph_node": "model"})]
    )
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
