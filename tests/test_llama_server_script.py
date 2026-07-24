import os
import subprocess
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "start-llama-server.sh"


def test_llama_server_script_requires_model_path() -> None:
    environment = os.environ.copy()
    environment.pop("MODEL_PATH", None)
    environment["ENV_FILE"] = "/missing/.env"

    result = subprocess.run(
        [str(SCRIPT_PATH)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "MODEL_PATH must be set" in result.stderr


def test_llama_server_script_rejects_missing_model_file() -> None:
    environment = {
        **os.environ,
        "ENV_FILE": "/missing/.env",
        "MODEL_PATH": "/missing/model.gguf",
    }

    result = subprocess.run(
        [str(SCRIPT_PATH)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "MODEL_PATH does not exist or is not a file" in result.stderr


def test_llama_server_script_loads_configured_env_file(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.touch()
    env_file = tmp_path / ".env"
    env_file.write_text(f"MODEL_PATH='{model_path}'\n")
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    llama_server_path = bin_path / "llama-server"
    llama_server_path.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    llama_server_path.chmod(0o755)
    environment = os.environ.copy()
    environment.pop("MODEL_PATH", None)
    environment["ENV_FILE"] = str(env_file)
    environment["PATH"] = f"{bin_path}:{environment['PATH']}"

    result = subprocess.run(
        [str(SCRIPT_PATH)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines()[:2] == ["--model", str(model_path)]


def test_llama_server_script_starts_server_with_configured_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.touch()
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    llama_server_path = bin_path / "llama-server"
    llama_server_path.write_text(
        "#!/bin/sh\n"
        "for argument in \"$@\"; do\n"
        "    echo \"$argument\"\n"
        "done\n"
    )
    llama_server_path.chmod(0o755)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    monkeypatch.setenv("LLAMA_SERVER_HOST", "0.0.0.0")
    monkeypatch.setenv("LLAMA_SERVER_PORT", "9090")
    monkeypatch.setenv("LLAMA_CONTEXT_SIZE", "4096")
    monkeypatch.setenv("LLAMA_GPU_LAYERS", "42")
    monkeypatch.setenv("ENV_FILE", str(tmp_path / "missing.env"))

    result = subprocess.run(
        [str(SCRIPT_PATH)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        "--model",
        str(model_path),
        "--host",
        "0.0.0.0",
        "--port",
        "9090",
        "--ctx-size",
        "4096",
        "--n-gpu-layers",
        "42",
        "--jinja",
        "--reasoning",
        "off",
    ]
