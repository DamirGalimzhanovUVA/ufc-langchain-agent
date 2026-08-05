from collections.abc import Iterator
from unittest.mock import Mock, call

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.errors import GraphRecursionError

import services.model_service as model_service_module
from services.model_service import (
    EmptyModelResponseError,
    ModelRefusalError,
    ModelService,
    convert_messages,
)


def test_initialize_creates_model_and_agent_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_model = Mock()
    chat_openai = Mock(return_value=chat_model)
    agent = Mock()
    create_agent = Mock(return_value=agent)
    news_call_limit = Mock()
    fight_description_call_limit = Mock()
    tool_call_limit = Mock(
        side_effect=[news_call_limit, fight_description_call_limit]
    )
    monkeypatch.setattr(model_service_module, "ChatOpenAI", chat_openai)
    monkeypatch.setattr(model_service_module, "create_agent", create_agent)
    monkeypatch.setattr(
        model_service_module, "ToolCallLimitMiddleware", tool_call_limit
    )
    monkeypatch.setenv("LLM_MODEL", "gpt-5-mini")
    service = ModelService()

    first_result = service.initialize()
    second_result = service.initialize()

    assert first_result is chat_model
    assert second_result is chat_model
    chat_openai.assert_called_once_with(
        model="gpt-5-mini",
        max_tokens=model_service_module.LLM_MAX_TOKENS,
    )
    assert service.get_model() is chat_model
    assert service.get_agent() is agent
    assert [tool.name for tool in service.tools] == [
        "fighter_news",
        "fighter_stats",
        "fighter_wikipedia",
        "fight_description",
    ]
    assert tool_call_limit.call_args_list == [
        call(
            tool_name="fighter_news",
            run_limit=3,
            exit_behavior="continue",
        ),
        call(
            tool_name="fight_description",
            run_limit=1,
            exit_behavior="continue",
        ),
    ]
    create_agent.assert_called_once_with(
        model=chat_model,
        tools=service.tools,
        system_prompt=model_service_module.SYSTEM_PROMPT,
        middleware=[
            model_service_module.log_news_search_decision,
            news_call_limit,
            fight_description_call_limit,
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


def test_system_prompt_limits_fight_description_to_one_call() -> None:
    assert "fight description tool no more than once" in (
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


def test_initialize_uses_gpt_5_nano_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_openai = Mock(return_value=Mock())
    monkeypatch.setattr(model_service_module, "ChatOpenAI", chat_openai)
    monkeypatch.setattr(model_service_module, "create_agent", Mock())
    monkeypatch.delenv("LLM_MODEL", raising=False)

    ModelService().initialize()

    chat_openai.assert_called_once_with(
        model="gpt-5-nano",
        max_tokens=model_service_module.LLM_MAX_TOKENS,
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
    ) -> Iterator[tuple[str, object]]:
        nonlocal stream_finished
        yield (
            "debug",
            {"step": 1, "type": "task", "payload": {"name": "model"}},
        )
        yield (
            "messages",
            (
                AIMessageChunk(
                    content="",
                    additional_kwargs={"reasoning_content": "Thinking "},
                ),
                {"langgraph_node": "model"},
            ),
        )
        yield (
            "messages",
            (
                AIMessageChunk(
                    content=[
                        {"type": "reasoning", "reasoning": "carefully. "}
                    ]
                ),
                {"langgraph_node": "model"},
            ),
        )
        yield (
            "messages",
            (
                AIMessageChunk(content="Final "),
                {"langgraph_node": "model"},
            ),
        )
        yield (
            "messages",
            (
                AIMessageChunk(content="answer"),
                {"langgraph_node": "model"},
            ),
        )
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
    assert logger.info.call_args_list == [
        call("Graph hop: step=%s node=%s", 1, "model"),
        call("Generated %s chunk: %r", "reasoning", "Thinking "),
    ]

    assert list(tokens) == ["carefully. ", "Final ", "answer"]
    assert stream_finished is True
    assert logger.info.call_args_list == [
        call("Graph hop: step=%s node=%s", 1, "model"),
        call("Generated %s chunk: %r", "reasoning", "Thinking "),
        call("Generated %s chunk: %r", "reasoning", "carefully. "),
        call("Generated %s chunk: %r", "text", "Final "),
        call("Generated %s chunk: %r", "text", "answer"),
        call(
            "Graph execution summary: supersteps=%s graph_tasks=%s "
            "model_calls=%s tool_calls=%s",
            1,
            1,
            1,
            0,
        ),
    ]
    invocation = agent.stream.call_args
    assert invocation.args[0] == {
        "messages": [HumanMessage(content="Question")]
    }
    assert invocation.kwargs == {
        "config": {"recursion_limit": 30},
        "stream_mode": ["messages", "debug"],
    }


def test_create_chat_completion_ignores_non_text_agent_events() -> None:
    agent = Mock()
    agent.stream.return_value = iter(
        [
            (
                "messages",
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
            ),
            (
                "messages",
                (
                    ToolMessage(
                        content='{"wins": 32}', tool_call_id="call-1"
                    ),
                    {"langgraph_node": "tools"},
                ),
            ),
            (
                "messages",
                (
                    AIMessageChunk(
                        content=[
                            {
                                "type": "text",
                                "text": "José Aldo has 32 wins.",
                            }
                        ]
                    ),
                    {"langgraph_node": "model"},
                ),
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


def test_create_chat_completion_requires_final_assistant_message() -> None:
    agent = Mock()
    agent.stream.return_value = iter(
        [
            (
                "messages",
                (
                    HumanMessage(content="Question"),
                    {"langgraph_node": "model"},
                ),
            )
        ]
    )
    service = ModelService()
    service.chat_model = Mock()
    service.agent = agent

    with pytest.raises(
        EmptyModelResponseError,
        match="model stream contained no assistant content",
    ):
        list(
            service.create_chat_completion(
                [{"role": "user", "content": "Question"}]
            )
        )


def test_create_chat_completion_captures_all_empty_model_response_chunks(
) -> None:
    tool_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": "fighter_stats",
                "args": '{"fighter_name":"José Aldo"}',
                "id": "call-1",
                "index": 0,
            }
        ],
    )
    terminal_chunk = AIMessageChunk(
        content="",
        response_metadata={"finish_reason": "length"},
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 2048,
            "total_tokens": 2148,
        },
    )
    agent = Mock()
    agent.stream.return_value = iter(
        [
            (
                "messages",
                (tool_chunk, {"langgraph_node": "model"}),
            ),
            (
                "messages",
                (terminal_chunk, {"langgraph_node": "model"}),
            ),
        ]
    )
    service = ModelService()
    service.chat_model = Mock()
    service.agent = agent

    with pytest.raises(
        EmptyModelResponseError,
        match="returned tool calls but no final assistant text",
    ) as raised_error:
        list(
            service.create_chat_completion(
                [{"role": "user", "content": "Question"}]
            )
        )

    assert raised_error.value.model_response is not None
    chunks = raised_error.value.model_response["chunks"]
    assert len(chunks) == 2
    assert chunks[0]["tool_call_chunks"][0]["name"] == "fighter_stats"
    assert chunks[1]["response_metadata"] == {
        "finish_reason": "length"
    }
    assert chunks[1]["usage_metadata"] == {
        "input_tokens": 100,
        "output_tokens": 2048,
        "total_tokens": 2148,
    }


@pytest.mark.parametrize(
    "chunk",
    [
        AIMessageChunk(
            content=[
                {
                    "type": "refusal",
                    "refusal": "I can't help with that request.",
                }
            ]
        ),
        AIMessageChunk(
            content="",
            additional_kwargs={
                "refusal": "I can't help with that request."
            },
        ),
    ],
)
def test_create_chat_completion_raises_model_refusal_error(
    chunk: AIMessageChunk,
) -> None:
    agent = Mock()
    agent.stream.return_value = iter(
        [
            (
                "messages",
                (chunk, {"langgraph_node": "model"}),
            )
        ]
    )
    service = ModelService()
    service.chat_model = Mock()
    service.agent = agent

    with pytest.raises(
        ModelRefusalError, match="model refused to answer the request"
    ):
        list(
            service.create_chat_completion(
                [{"role": "user", "content": "Question"}]
            )
        )


def test_create_chat_completion_logs_graph_summary_on_recursion_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stream(
        *args: object, **kwargs: object
    ) -> Iterator[tuple[str, object]]:
        yield (
            "debug",
            {"step": 1, "type": "task", "payload": {"name": "model"}},
        )
        yield (
            "debug",
            {
                "step": 2,
                "type": "task",
                "payload": {
                    "name": "ToolCallLimitMiddleware[fighter_news].after_model"
                },
            },
        )
        yield (
            "debug",
            {"step": 3, "type": "task", "payload": {"name": "tools"}},
        )
        yield (
            "debug",
            {"step": 4, "type": "task", "payload": {"name": "model"}},
        )
        raise GraphRecursionError("Recursion limit reached")

    agent = Mock()
    agent.stream.side_effect = stream
    logger = Mock()
    monkeypatch.setattr(model_service_module, "logger", logger)
    service = ModelService()
    service.chat_model = Mock()
    service.agent = agent

    with pytest.raises(GraphRecursionError, match="Recursion limit reached"):
        list(
            service.create_chat_completion(
                [{"role": "user", "content": "Question"}]
            )
        )

    assert logger.info.call_args_list[-1] == call(
        "Graph execution summary: supersteps=%s graph_tasks=%s "
        "model_calls=%s tool_calls=%s",
        4,
        4,
        2,
        1,
    )
