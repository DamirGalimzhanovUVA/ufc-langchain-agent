import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from services.model_service import ModelService, model_service


def load_environment() -> None:
    project_root = Path(__file__).resolve().parents[2]
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


@app.post("/chat", response_class=StreamingResponse)
def create_chat_completion(
    chat_request: ChatRequest, request: Request
) -> StreamingResponse:
    service: ModelService = request.app.state.model_service
    messages = [message.model_dump() for message in chat_request.messages]
    tokens = service.create_chat_completion(messages)
    return StreamingResponse(tokens, media_type="text/plain")
