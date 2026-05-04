import os
from fastapi import APIRouter, BackgroundTasks, File, Form, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.services.ingestion_service import process_uploaded_file, process_uploaded_url
from app.schemas.responses import UploadResponse


router = APIRouter()
REDIS_URL = os.getenv("REDIS_URL")

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=REDIS_URL if REDIS_URL else None
)


# Rate limit: 10 uploads per minute per IP Address
# File upload endpoint
@router.post("/upload-file", response_model=UploadResponse)
@limiter.limit("10/minute")
async def upload_file(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    return await process_uploaded_file(file, background_tasks)


# URL upload endpoint
@router.post("/upload-url", response_model=UploadResponse)
@limiter.limit("10/minute")
async def upload_url(request: Request, background_tasks: BackgroundTasks, url: str = Form(...)):
    return await process_uploaded_url(url, background_tasks)
