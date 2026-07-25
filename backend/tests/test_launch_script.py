import os
import subprocess
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[2] / "launch.sh"


def create_command(
    bin_dir: Path,
    command_name: str,
    output_file: Path,
    run_seconds: float,
) -> None:
    command_path = bin_dir / command_name
    command_path.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" '
        f'"$PWD|$BACKEND_HOST|$BACKEND_PORT|$FRONTEND_HOST|'
        f'$FRONTEND_PORT|$ENV_FILE|$PYTHONPATH|$VITE_API_TARGET|'
        f'$NODE_ENV|$*" > "{output_file}"\n'
        f"sleep {run_seconds}\n"
    )
    command_path.chmod(0o755)


def test_launch_script_sets_service_environments(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    backend_output = tmp_path / "backend-output"
    frontend_output = tmp_path / "frontend-output"
    model_output = tmp_path / "model-output"
    model_path = tmp_path / "model.gguf"
    model_path.touch()
    create_command(bin_dir, "uvicorn", backend_output, 1)
    create_command(bin_dir, "npm", frontend_output, 1)
    create_command(bin_dir, "llama-server", model_output, 0.1)
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "BACKEND_HOST": "127.0.0.2",
        "BACKEND_PORT": "9000",
        "BACKEND_ENV_FILE": "/run/config/backend.env",
        "FRONTEND_HOST": "127.0.0.3",
        "FRONTEND_PORT": "6000",
        "MODEL_PATH": str(model_path),
        "LLAMA_SERVER_HOST": "127.0.0.4",
        "LLAMA_SERVER_PORT": "7000",
        "LLAMA_CONTEXT_SIZE": "4096",
        "LLAMA_GPU_LAYERS": "0",
        "PYTHONPATH": "",
    }

    result = subprocess.run(
        [str(SCRIPT_PATH)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    project_dir = SCRIPT_PATH.parent
    assert result.returncode == 0
    assert backend_output.read_text().strip() == (
        f"{project_dir / 'backend/app'}|127.0.0.2|9000|127.0.0.3|6000|"
        f"/run/config/backend.env|{project_dir / 'backend/app'}|"
        "http://127.0.0.1:9000|development|"
        "main:app --host 127.0.0.2 --port 9000"
    )
    assert frontend_output.read_text().strip() == (
        f"{project_dir / 'frontend'}|127.0.0.2|9000|127.0.0.3|6000|"
        f"/run/config/backend.env|{project_dir / 'backend/app'}|"
        "http://127.0.0.1:9000|development|"
        "run dev -- --host 127.0.0.3 --port 6000"
    )
    assert model_output.read_text().strip() == (
        f"{project_dir}|127.0.0.2|9000|127.0.0.3|6000|"
        f"/run/config/backend.env|{project_dir / 'backend/app'}|"
        "http://127.0.0.1:9000|development|"
        f"--model {model_path} --host 127.0.0.4 --port 7000 "
        "--ctx-size 4096 --n-gpu-layers 0 --jinja --reasoning off"
    )
