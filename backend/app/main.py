import logging
import os
from contextlib import asynccontextmanager

import rag_store
from database import engine
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import Base
from routers import auth, rag
from sqlalchemy import text

logger = logging.getLogger(__name__)


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # DB tables
    Base.metadata.create_all(bind=engine)

    # Optional: build the vector DB from local JSON files at startup
    if _truthy(os.getenv("RAG_AUTO_INDEX", "false")):
        docs_dir = os.getenv("DOCS_DIR", "/data/docs")
        rebuild = _truthy(os.getenv("RAG_REBUILD_ON_START", "true"))
        fail_fast = _truthy(os.getenv("RAG_INDEX_FAIL_FAST", "false"))

        try:
            if rebuild:
                stats = rag_store.rebuild_from_json_dir(docs_dir)
            else:
                stats = rag_store.ingest_json_dir(docs_dir)
            logger.info("RAG index ready: %s", stats)
        except Exception as exc:
            logger.exception("RAG auto-index failed: %s", exc)
            if fail_fast:
                raise

    yield


app = FastAPI(title="APIs", lifespan=lifespan)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    with engine.connect() as conn:
        ok = conn.execute(text("SELECT 1")).scalar() == 1
    return {"status": "ok", "db": ok}


app.include_router(auth.router)
app.include_router(rag.router)
