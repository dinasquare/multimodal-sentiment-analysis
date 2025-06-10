from sqlalchemy.orm import Session
from ..models.db_models import Job, JobStatus
from typing import Optional


class ProgressTracker:
    """Utility class for tracking job progress."""
    
    def __init__(self, job_id: str, db: Session):
        self.job_id = job_id
        self.db = db
        self.current_step_num = 0
        self.total_steps = 7
        
    def update_progress(self, 
                       step_name: str, 
                       step_details: Optional[str] = None, 
                       progress_percentage: Optional[float] = None):
        """Update job progress in database."""
        try:
            job = self.db.query(Job).filter(Job.id == self.job_id).first()
            if not job:
                return
                
            job.current_step = step_name
            if step_details:
                job.step_details = step_details
                
            if progress_percentage is not None:
                job.progress_percentage = progress_percentage
            else:
                # Auto-calculate based on step number
                job.progress_percentage = (self.current_step_num / self.total_steps) * 100
                
            self.db.commit()
            print(f"Progress: {job.progress_percentage:.1f}% - {step_name}")
            if step_details:
                print(f"Details: {step_details}")
                
        except Exception as e:
            print(f"Error updating progress: {str(e)}")
    
    def next_step(self, step_name: str, step_details: Optional[str] = None):
        """Move to next step and update progress."""
        self.current_step_num += 1
        progress = (self.current_step_num / self.total_steps) * 100
        self.update_progress(step_name, step_details, progress)
    
    def set_error(self, error_message: str):
        """Set job as failed with error message."""
        try:
            job = self.db.query(Job).filter(Job.id == self.job_id).first()
            if job:
                job.status = JobStatus.FAILED
                job.error_message = error_message
                job.current_step = "Failed"
                job.step_details = error_message
                self.db.commit()
        except Exception as e:
            print(f"Error setting job error: {str(e)}")
    
    def set_completed(self):
        """Mark job as completed."""
        try:
            job = self.db.query(Job).filter(Job.id == self.job_id).first()
            if job:
                job.status = JobStatus.COMPLETED
                job.progress_percentage = 100.0
                job.current_step = "Completed"
                job.step_details = "Sentiment analysis completed successfully"
                self.db.commit()
        except Exception as e:
            print(f"Error setting job completed: {str(e)}") 