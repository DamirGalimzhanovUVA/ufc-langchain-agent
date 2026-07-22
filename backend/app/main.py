import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel

from services.model_service import ModelService, model_service


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


class ChatResponse(BaseModel):
    response: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_path = os.environ.get("LLAMA_MODEL_PATH")
    if model_path is None:
        raise RuntimeError("LLAMA_MODEL_PATH must be set")

    model_service.initialize(model_path)
    app.state.model_service = model_service
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/chat", response_model=ChatResponse)
def create_chat_completion(chat_request: ChatRequest, request: Request) -> ChatResponse:
    service: ModelService = request.app.state.model_service
    messages = [message.model_dump() for message in chat_request.messages]
    response = service.create_chat_completion(messages)
    return ChatResponse(response=response)
