import asyncio
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import FastAPI

import main


def test_load_environment_reads_default_backend_env_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_dotenv = Mock()
    monkeypatch.setattr(main, "load_dotenv", load_dotenv)
    monkeypatch.delenv("ENV_FILE", raising=False)

    main.load_environment()

    expected_env_file = Path(main.__file__).resolve().parents[1] / ".env"
    load_dotenv.assert_called_once_with(expected_env_file)


def test_load_environment_reads_configured_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=test-model\n")
    monkeypatch.setenv("ENV_FILE", str(env_file))
    monkeypatch.delenv("LLM_MODEL", raising=False)

    main.load_environment()

    assert os.environ["LLM_MODEL"] == "test-model"
    monkeypatch.delenv("LLM_MODEL")


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
