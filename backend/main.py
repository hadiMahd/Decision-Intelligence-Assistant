import uvicorn
from fastapi import FastAPI

from config import ensure_artifact_dirs
from routers.ml_router import router as ml_router
from routers.qdrant_db import router as qdrant_router
from routers.rag_router import router as rag_router


app = FastAPI(title="Support Ticket Compare API", version="0.1.0")

app.include_router(rag_router)
app.include_router(ml_router)
app.include_router(qdrant_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def run() -> None:
    ensure_artifact_dirs()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
