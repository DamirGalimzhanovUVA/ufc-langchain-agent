import asyncio
from unittest.mock import Mock

import pytest
from fastapi import FastAPI

import main


def test_lifespan_initializes_and_stores_model_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_service = Mock()
    monkeypatch.setattr(main, "model_service", model_service)
    test_app = FastAPI()

    async def run_lifespan() -> None:
        async with main.lifespan(test_app):
            assert test_app.state.model_service is model_service

    asyncio.run(run_lifespan())

    model_service.initialize.assert_called_once_with()
