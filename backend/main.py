import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import ensure_artifact_dirs
from routers.ml_router import router as ml_router
from routers.qdrant_db import router as qdrant_router
from routers.rag_router import router as rag_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(title="Support Ticket Compare API", version="0.1.0")

# Enable CORS for browser-based frontend requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router)
app.include_router(ml_router)
app.include_router(qdrant_router)


@app.get("/health")
def health() -> dict:
    logger.info("Health check endpoint called")
    return {"status": "ok"}


def run() -> None:
    ensure_artifact_dirs()
    logger.info("Starting FastAPI server on http://0.0.0.0:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    run()
