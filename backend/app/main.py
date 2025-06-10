from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
import os
from contextlib import asynccontextmanager

from .api.api import router as api_router
from .models.db_models import create_tables
from .processing.job_processor import start_processing_thread, stop_processing_thread

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle events for FastAPI application.
    """
    # Create database tables
    create_tables()
    logger.info("Database tables created")
    
    # Start job processing thread
    start_processing_thread()
    logger.info("Job processing thread started")
    
    yield
    
    # Stop job processing thread
    stop_processing_thread()
    logger.info("Job processing thread stopped")


# Create FastAPI application
app = FastAPI(
    title="Multi-Modal Sentiment Analysis API",
    description="API for analyzing sentiment in videos through visual, audio, and text modalities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add middleware for request logging and timing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and their processing time."""
    start_time = time.time()
    
    response = await call_next(request)
    
    # Log request details
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} completed in {process_time:.4f}s")
    
    return response


# Add exception handler for unexpected errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.exception(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


# Include API router
app.include_router(api_router, prefix="/api/v1")


# Root endpoint
@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Multi-Modal Sentiment Analysis API",
        "version": "1.0.0",
        "status": "active"
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"} 