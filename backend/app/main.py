import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.services.model_service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_path = os.environ.get("LLAMA_MODEL_PATH")
    if model_path is None:
        raise RuntimeError("LLAMA_MODEL_PATH must be set")

    model_service.initialize(model_path)
    app.state.model_service = model_service
    yield


app = FastAPI(lifespan=lifespan)
