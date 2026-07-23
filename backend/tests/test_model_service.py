import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

import services.model_service as model_service_module
from services.model_service import LlamaCppChatModel, ModelService


def test_initialize_creates_model_once(monkeypatch: pytest.MonkeyPatch) -> None:
    model = Mock()
    llama = Mock(return_value=model)
    monkeypatch.setattr(model_service_module, "Llama", llama)
    service = ModelService()

    first_result = service.initialize("first-model.gguf")
    second_result = service.initialize("second-model.gguf")

    assert first_result is second_result
    assert first_result.model is model
    llama.assert_called_once_with(
        model_path="first-model.gguf",
        n_ctx=8192,
        n_gpu_layers=-1,
        verbose=True,
    )
    assert service.get_model() is first_result
    assert [tool.name for tool in service.tools] == [
        "fighter_news",
        "fighter_stats",
        "fighter_wikipedia",
    ]


def test_get_model_returns_initialized_model(monkeypatch: pytest.MonkeyPatch) -> None:
    model = Mock()
    monkeypatch.setattr(model_service_module, "Llama", Mock(return_value=model))
    service = ModelService()

    service.initialize("model.gguf")

    assert service.get_model().model is model


def test_get_model_raises_when_model_is_not_initialized() -> None:
    service = ModelService()

    with pytest.raises(RuntimeError, match="The model has not been initialized"):
        service.get_model()


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


def test_llama_cpp_chat_model_parses_streamed_tool_call() -> None:
    model = Mock()
    model.create_chat_completion.return_value = iter(
        [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "fighter_stats",
                                        "arguments": '{"fighter_name":',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": '"Amanda Nunes"}'
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
        ]
    )
    chat_model = LlamaCppChatModel(model=model)

    response = chat_model.invoke(
        [HumanMessage("What is Amanda Nunes's record?")],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "fighter_stats",
                    "description": "Get fighter statistics.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fighter_name": {"type": "string"}
                        },
                        "required": ["fighter_name"],
                    },
                },
            }
        ],
    )

    assert response.tool_calls == [
        {
            "name": "fighter_stats",
            "args": {"fighter_name": "Amanda Nunes"},
            "id": "call-1",
            "type": "tool_call",
        }
    ]


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


def test_create_chat_completion_registers_and_executes_tools() -> None:
    def get_fighter_stats(fighter_name: str) -> dict[str, object]:
        """Get fighter statistics."""
        return {"fighter": fighter_name, "wins": 30}

    tool = StructuredTool.from_function(
        get_fighter_stats, name="fighter_stats"
    )
    bound_model = Mock()
    bound_model.stream.side_effect = [
        iter(
            [
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
                )
            ]
        ),
        iter(
            [
                AIMessageChunk(content="José Aldo has "),
                AIMessageChunk(content="30 wins."),
            ]
        ),
    ]
    chat_model = Mock()
    chat_model.bind_tools.return_value = bound_model
    service = ModelService()
    service.chat_model = chat_model
    service.tools = [tool]
    messages = [{"role": "user", "content": "How many wins does José Aldo have?"}]

    tokens = list(service.create_chat_completion(messages))

    assert tokens == ["José Aldo has ", "30 wins."]
    chat_model.bind_tools.assert_called_once_with([tool])
    first_messages = bound_model.stream.call_args_list[0].args[0]
    second_messages = bound_model.stream.call_args_list[1].args[0]
    assert first_messages == messages
    assert second_messages[-1].tool_call_id == "call-1"
    assert json.loads(second_messages[-1].content) == {
        "fighter": "José Aldo",
        "wins": 30,
    }
