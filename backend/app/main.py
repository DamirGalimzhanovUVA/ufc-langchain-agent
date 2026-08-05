import json
import logging
import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from starlette.responses import JSONResponse, Response, StreamingResponse

from services.model_service import (
    ModelRefusalError,
    ModelResponseError,
    ModelService,
    model_service,
)


logger = logging.getLogger("uvicorn.error")
SHOW_MODEL_RESPONSE_JSON = True
CHAT_ERROR_MESSAGE = (
    "We couldn't complete your request. Please send your message again."
)
CHAT_REFUSAL_MESSAGE = (
    "The model couldn't answer that request. Try rephrasing your message."
)


def load_environment() -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_env_file = project_root / ".env"
    env_file = Path(os.getenv("ENV_FILE", str(default_env_file)))
    load_dotenv(env_file)


load_environment()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_service.initialize()
    app.state.model_service = model_service
    yield


app = FastAPI(lifespan=lifespan)


def get_error_content(
    message: str,
    retryable: bool,
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "retryable": retryable,
        }
    }


def log_chat_exception(message: str, error: Exception) -> None:
    if (
        SHOW_MODEL_RESPONSE_JSON
        and isinstance(error, ModelResponseError)
        and error.model_response is not None
    ):
        logger.exception(
            "%s\nModel response JSON:\n%s",
            message,
            json.dumps(error.model_response, indent=2),
        )
        return

    logger.exception(message)


def stream_chat_events(tokens: Iterator[str]) -> Iterator[str]:
    try:
        for token in tokens:
            yield json.dumps({"content": token}) + "\n"
    except ModelRefusalError as error:
        log_chat_exception(
            "Chat completion was refused while streaming",
            error,
        )
        yield json.dumps(
            get_error_content(CHAT_REFUSAL_MESSAGE, False)
        ) + "\n"
    except Exception as error:
        log_chat_exception("Chat completion failed while streaming", error)
        yield json.dumps(
            get_error_content(CHAT_ERROR_MESSAGE, True)
        ) + "\n"


@app.post("/chat")
def create_chat_completion(
    chat_request: ChatRequest, request: Request
) -> Response:
    service: ModelService = request.app.state.model_service
    messages = [message.model_dump() for message in chat_request.messages]
    try:
        tokens = service.create_chat_completion(messages)
    except ModelRefusalError as error:
        log_chat_exception(
            "Chat completion was refused before streaming",
            error,
        )
        return JSONResponse(
            status_code=400,
            content=get_error_content(CHAT_REFUSAL_MESSAGE, False),
        )
    except Exception as error:
        log_chat_exception("Chat completion failed before streaming", error)
        return JSONResponse(
            status_code=500,
            content=get_error_content(CHAT_ERROR_MESSAGE, True),
        )

    return StreamingResponse(
        stream_chat_events(tokens),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
