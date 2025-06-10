from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import uuid
import json
from typing import Optional
from ..models.db_models import get_db, Job, JobStatus
from ..models.schemas import JobStatusResponse, JobResult, SentimentResult
from ..utils.file_handler import save_upload_file, cleanup_job_files
from ..processing.simple_processor import get_processor
import os


router = APIRouter()


@router.post("/upload", response_model=JobResult)
async def upload_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload and immediately process a video file for sentiment analysis.
    
    Args:
        file: The video file to upload
        db: Database session
    
    Returns:
        JobResult: Complete job results with sentiment analysis
    """
    job_id = None
    try:
        # Save uploaded file
        job_id, file_path = await save_upload_file(file)
        
        # Create job in database
        job = Job(
            id=job_id,
            original_filename=file.filename,
            file_path=file_path,
            status=JobStatus.PROCESSING
        )
        db.add(job)
        db.commit()
        
        # Process video directly using local models
        processor = get_processor()
        results = processor.process_video(file_path)
        
        if "error" in results:
            # Processing failed
            job.status = JobStatus.FAILED
            job.error_message = results["error"]
            db.commit()
            
            # Clean up files
            cleanup_job_files(job_id)
            
            raise HTTPException(status_code=500, detail=results["error"])
        
        # Processing successful - update job with results
        job.status = JobStatus.COMPLETED
        job.visual_sentiment = results.get("visual_sentiment")
        job.audio_sentiment = results.get("audio_sentiment") 
        job.text_sentiment = results.get("text_sentiment")
        job.combined_sentiment = results.get("combined_sentiment")
        job.confidence = results.get("confidence")
        db.commit()
        
        # Clean up uploaded file (keep only if needed for debugging)
        cleanup_job_files(job_id)
        
        # Return complete results with additional data
        sentiment_result = SentimentResult(
            visual_sentiment=job.visual_sentiment,
            audio_sentiment=job.audio_sentiment,
            text_sentiment=job.text_sentiment,
            combined_sentiment=job.combined_sentiment,
            confidence=job.confidence,
            transcription=results.get("transcription"),
            visual_emotions=results.get("visual_emotions"),
            dominant_visual_emotion=results.get("dominant_visual_emotion")
        )
        
        return JobResult(
            id=job.id,
            original_filename=job.original_filename,
            created_at=job.created_at,
            updated_at=job.updated_at,
            status=job.status,
            results=sentiment_result,
            error_message=job.error_message
        )
    
    except Exception as e:
        # Clean up any uploaded files in case of error
        if job_id:
            try:
                cleanup_job_files(job_id)
            except:
                pass
        
        # Update job status if it exists
        if job_id:
            try:
                job = db.query(Job).filter(Job.id == job_id).first()
                if job:
                    job.status = JobStatus.FAILED
                    job.error_message = str(e)
                    db.commit()
            except:
                pass
        
        # Re-raise exception
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """
    Get the status of a sentiment analysis job.
    
    Args:
        job_id: Job ID to check
        db: Database session
    
    Returns:
        JobStatusResponse: Job status information
    """
    # Get job from database
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    # Return job status (simplified - no complex progress tracking)
    return JobStatusResponse(
        id=job.id,
        status=job.status,
        error_message=job.error_message,
        progress_percentage=100.0 if job.status == JobStatus.COMPLETED else 0.0,
        current_step="Completed" if job.status == JobStatus.COMPLETED else "Processing",
        step_details=None,
        total_steps=1
    )


@router.get("/results/{job_id}", response_model=JobResult)
async def get_job_results(job_id: str, db: Session = Depends(get_db)):
    """
    Get the results of a completed sentiment analysis job.
    
    Args:
        job_id: Job ID to get results for
        db: Database session
    
    Returns:
        JobResult: Job results information
    """
    # Get job from database
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    if job.status == JobStatus.PROCESSING:
        raise HTTPException(status_code=202, detail=f"Job is still processing: {job.status}")
    
    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=500, detail=f"Job failed: {job.error_message}")
    
    # Construct results
    results = None
    if job.status == JobStatus.COMPLETED:
        results = SentimentResult(
            visual_sentiment=job.visual_sentiment,
            audio_sentiment=job.audio_sentiment,
            text_sentiment=job.text_sentiment,
            combined_sentiment=job.combined_sentiment,
            confidence=job.confidence
        )
    
    # Return job result
    return JobResult(
        id=job.id,
        original_filename=job.original_filename,
        created_at=job.created_at,
        updated_at=job.updated_at,
        status=job.status,
        results=results,
        error_message=job.error_message
    )


@router.delete("/cleanup/{job_id}")
async def cleanup_job(job_id: str, db: Session = Depends(get_db)):
    """
    Clean up files associated with a job.
    
    Args:
        job_id: Job ID to clean up
        db: Database session
    
    Returns:
        dict: Status message
    """
    # Get job from database
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    # Clean up files
    cleanup_job_files(job_id)
    
    return {"message": f"Files for job {job_id} have been cleaned up"} 