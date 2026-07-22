import asyncio
from unittest.mock import Mock

import pytest
from fastapi import FastAPI

import main


def test_lifespan_initializes_and_stores_model_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_service = Mock()
    monkeypatch.setenv("LLAMA_MODEL_PATH", "model.gguf")
    monkeypatch.setattr(main, "model_service", model_service)
    test_app = FastAPI()

    async def run_lifespan() -> None:
        async with main.lifespan(test_app):
            assert test_app.state.model_service is model_service

    asyncio.run(run_lifespan())

    model_service.initialize.assert_called_once_with("model.gguf")


def test_lifespan_raises_when_model_path_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLAMA_MODEL_PATH", raising=False)
    test_app = FastAPI()

    async def run_lifespan() -> None:
        async with main.lifespan(test_app):
            pass

    with pytest.raises(RuntimeError, match="LLAMA_MODEL_PATH must be set"):
        asyncio.run(run_lifespan())
