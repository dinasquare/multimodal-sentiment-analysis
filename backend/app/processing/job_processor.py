import asyncio
import threading
from pathlib import Path
import time
from sqlalchemy.orm import Session
from ..models.db_models import Job, JobStatus
from ..utils.file_handler import cleanup_job_files
from ..utils.progress_tracker import ProgressTracker
from .video_processor import extract_audio_from_video, extract_frames_from_video
from .sentiment_analyzer import (
    analyze_visual_sentiment, 
    analyze_audio_sentiment, 
    analyze_text_sentiment,
    transcribe_audio,
    fusion_algorithm
)


# Job queue
job_queue = asyncio.Queue()
# Processing thread
processing_thread = None
# Flag to control thread execution
should_continue = True


async def process_job(job_id: str, db: Session):
    """
    Process a sentiment analysis job.
    
    Args:
        job_id: Job ID to process
        db: Database session
    """
    progress_tracker = ProgressTracker(job_id, db)
    
    try:
        # Get job from database
        job = db.query(Job).filter(Job.id == job_id).first()
        
        if not job:
            print(f"Job {job_id} not found")
            return
        
        # Update job status
        job.status = JobStatus.PROCESSING
        db.commit()
        
        progress_tracker.next_step("Starting Processing", "Initializing video analysis...")
        
        # Get video path
        video_path = job.file_path
        
        progress_tracker.next_step("Extracting Frames", "Extracting frames from video...")
        # Extract frames from video
        frame_paths = extract_frames_from_video(video_path, job_id)
        
        progress_tracker.next_step("Extracting Audio", "Extracting audio from video...")
        # Extract audio from video
        audio_path = extract_audio_from_video(video_path, job_id)
        
        progress_tracker.next_step("Analyzing Visual Content", "Processing facial emotions...")
        # Process visual modality
        visual_result = analyze_visual_sentiment(frame_paths, job_id, db)
        
        progress_tracker.next_step("Transcribing Audio", "Converting speech to text...")
        # Transcribe audio to text
        text = transcribe_audio(audio_path, job_id, db)
        
        progress_tracker.next_step("Analyzing Audio", "Processing audio sentiment...")
        # Process audio modality
        audio_result = analyze_audio_sentiment(audio_path, job_id, db)
        
        progress_tracker.next_step("Analyzing Text", "Processing text sentiment...")
        # Process text modality
        text_result = analyze_text_sentiment(text, job_id, db)
        
        progress_tracker.update_progress("Combining Results", "Fusing multi-modal analysis...")
        # Combine results using fusion algorithm
        fusion_result = fusion_algorithm(visual_result, audio_result, text_result)
        
        # Update job with results
        job.visual_sentiment = visual_result["sentiment_score"]
        job.audio_sentiment = audio_result["sentiment_score"]
        job.text_sentiment = text_result["sentiment_score"]
        job.combined_sentiment = fusion_result["combined_sentiment"]
        job.confidence = fusion_result["confidence"]
        
        # Mark as completed
        progress_tracker.set_completed()
        
    except Exception as e:
        error_msg = f"Error processing job {job_id}: {str(e)}"
        print(error_msg)
        progress_tracker.set_error(error_msg)


async def job_processor_loop():
    """Background job processor loop."""
    from ..models.db_models import SessionLocal
    
    while should_continue:
        try:
            # Get job from queue
            job_id = await job_queue.get()
            
            # Create database session
            db = SessionLocal()
            try:
                # Process job
                await process_job(job_id, db)
            finally:
                db.close()
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in job processor: {str(e)}")
            await asyncio.sleep(5)  # Wait a bit before retrying
            
        # Mark task as done
        job_queue.task_done()


def start_processing_thread():
    """Start the background processing thread."""
    global processing_thread, should_continue
    
    if processing_thread and processing_thread.is_alive():
        return
        
    should_continue = True
    
    async def start_loop():
        await job_processor_loop()
    
    def thread_target():
        asyncio.run(start_loop())
    
    processing_thread = threading.Thread(target=thread_target, daemon=True)
    processing_thread.start()


def stop_processing_thread():
    """Stop the background processing thread."""
    global should_continue
    should_continue = False
    
    if processing_thread and processing_thread.is_alive():
        processing_thread.join(timeout=5)


async def submit_job(job_id: str):
    """
    Submit a job to the processing queue.
    
    Args:
        job_id: Job ID to process
    """
    await job_queue.put(job_id) 