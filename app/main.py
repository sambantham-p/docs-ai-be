from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from app.core.logger import setup_logging
from app.db.faiss_store import force_save_index
from app.db.mongodb import setup_indexes
from app.utils.chunker_utils import warmup_tokenizer
from app.routers.upload import router as upload_router
from app.routers.query import router as query_router
from app.routers.qa import router as qa_router
from app.utils.http_client_util import http_client

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting DocMind Application...")
    logger.info("Loading tokenizer...")
    warmup_tokenizer()
    logger.info("Tokenizer ready.")
    # Mongo indexes
    logger.info("Ensuring MongoDB indexes...")
    logger.info("Startup complete")
    await setup_indexes()
    yield
    force_save_index()
    await http_client.aclose()
    logger.info("Shutting down DocMind Application...")


# App instance
app = FastAPI(
    title="DocMind API",
    description="Multi document Q&A system",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routers
app.include_router(upload_router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(query_router, prefix="/api/v1", tags=["Query"])
app.include_router(qa_router, prefix="/api/v1", tags=["QA"])



# Health check
@app.get("/health", tags=["System"])
def health():
    logger.info("Health check called")
    return {
        "status": "ok",
        "service": "DocMind",
        "version": "1.0.0",
    }


# Root
@app.get("/", tags=["System"])
def root():
    logger.info("Root endpoint accessed")
    return {
        "message": "DocMind API is running",
        "docs": "/docs",
        "health": "/health",
    }