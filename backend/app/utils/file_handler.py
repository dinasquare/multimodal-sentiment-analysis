import os
import shutil
import uuid
from pathlib import Path
from fastapi import UploadFile, HTTPException
import aiofiles

# Define upload directory
UPLOAD_DIR = Path("uploads")
# Define temporary work directory
TEMP_DIR = Path("temp")

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Allowed video formats
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov"}
# Maximum file size (100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB in bytes


def get_extension(filename: str) -> str:
    """Get file extension from filename."""
    return filename.split(".")[-1] if "." in filename else ""


def is_valid_video(filename: str) -> bool:
    """Check if the file has an allowed video extension."""
    return get_extension(filename).lower() in ALLOWED_EXTENSIONS


async def save_upload_file(file: UploadFile) -> tuple[str, str]:
    """
    Save an uploaded video file.
    
    Args:
        file: The uploaded file
        
    Returns:
        tuple: (file_id, file_path)
        
    Raises:
        HTTPException: If file is invalid or too large
    """
    # Verify file is a video
    if not is_valid_video(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Generate unique ID and create directory
    file_id = str(uuid.uuid4())
    file_dir = UPLOAD_DIR / file_id
    file_dir.mkdir(exist_ok=True)
    
    # Save file with original extension
    extension = get_extension(file.filename)
    file_path = file_dir / f"original.{extension}"
    
    # Check file size as we save
    size = 0
    async with aiofiles.open(file_path, "wb") as f:
        # Read and write the file in chunks to avoid memory issues
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                # Clean up and raise error
                if file_path.exists():
                    file_path.unlink()
                if file_dir.exists() and file_dir.is_dir():
                    shutil.rmtree(file_dir)
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)}MB"
                )
            await f.write(chunk)
    
    return file_id, str(file_path)


def get_temp_dir(job_id: str) -> Path:
    """Get temporary directory for job processing."""
    temp_dir = TEMP_DIR / job_id
    temp_dir.mkdir(exist_ok=True)
    return temp_dir


def cleanup_job_files(job_id: str) -> None:
    """Clean up all files related to a job."""
    # Clean up upload directory
    upload_dir = UPLOAD_DIR / job_id
    if upload_dir.exists() and upload_dir.is_dir():
        shutil.rmtree(upload_dir)
    
    # Clean up temp directory
    temp_dir = TEMP_DIR / job_id
    if temp_dir.exists() and temp_dir.is_dir():
        shutil.rmtree(temp_dir) 