import json
import logging
import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from starlette.responses import JSONResponse, Response, StreamingResponse

from services.model_service import ModelService, model_service


logger = logging.getLogger("uvicorn.error")
CHAT_ERROR_MESSAGE = (
    "We couldn't complete your request. Please send your message again."
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


def stream_chat_events(tokens: Iterator[str]) -> Iterator[str]:
    try:
        for token in tokens:
            yield json.dumps({"content": token}) + "\n"
    except Exception:
        logger.exception("Chat completion failed while streaming")
        yield json.dumps(
            {
                "error": {
                    "message": CHAT_ERROR_MESSAGE,
                    "retryable": True,
                }
            }
        ) + "\n"


@app.post("/chat")
def create_chat_completion(
    chat_request: ChatRequest, request: Request
) -> Response:
    service: ModelService = request.app.state.model_service
    messages = [message.model_dump() for message in chat_request.messages]
    try:
        tokens = service.create_chat_completion(messages)
    except Exception:
        logger.exception("Chat completion failed before streaming")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": CHAT_ERROR_MESSAGE,
                    "retryable": True,
                }
            },
        )

    return StreamingResponse(
        stream_chat_events(tokens),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
