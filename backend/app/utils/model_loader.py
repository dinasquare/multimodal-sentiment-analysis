import os
import torch
from pathlib import Path
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification, 
    AutoModel,
    WhisperProcessor,
    WhisperForConditionalGeneration
)
from functools import lru_cache
from typing import Optional, Callable
from sqlalchemy.orm import Session
import requests
from tqdm import tqdm


class ModelLoader:
    """Utility class for loading and caching AI models."""
    
    _models = {}
    _processors = {}
    
    # Local models directory
    MODELS_DIR = Path("models")
    
    @classmethod
    def get_local_model_path(cls, model_type: str) -> Optional[Path]:
        """Get local model path if it exists."""
        model_paths = {
            "visual": cls.MODELS_DIR / "visual_emotion_detection",
            "text": cls.MODELS_DIR / "text_sentiment", 
            "audio": cls.MODELS_DIR / "audio_sentiment",
            "whisper": cls.MODELS_DIR / "whisper_base"
        }
        
        local_path = model_paths.get(model_type)
        if local_path and local_path.exists() and (local_path / "config.json").exists():
            return local_path
        return None
    
    @classmethod
    def update_progress(cls, job_id: str, db: Session, step: str, details: str, progress: Optional[float] = None):
        """Update progress in database if job_id and db are provided."""
        if job_id and db:
            try:
                from ..models.db_models import Job
                job = db.query(Job).filter(Job.id == job_id).first()
                if job:
                    job.current_step = step
                    job.step_details = details
                    if progress is not None:
                        job.progress_percentage = progress
                    db.commit()
                    print(f"Model Loading: {step} - {details}")
                    if progress is not None:
                        print(f"Progress: {progress:.1f}%")
            except Exception as e:
                print(f"Error updating model loading progress: {str(e)}")
    
    @classmethod
    def download_with_progress(cls, model_name: str, model_type: str, job_id: Optional[str] = None, db: Optional[Session] = None):
        """Download model with progress tracking."""
        if job_id and db:
            cls.update_progress(
                job_id, db, 
                f"Downloading {model_type}", 
                f"Downloading {model_name} from Hugging Face...",
                0
            )
        
        # Create a custom progress callback
        def progress_callback(current: int, total: int):
            if job_id and db and total > 0:
                progress = (current / total) * 100
                cls.update_progress(
                    job_id, db,
                    f"Downloading {model_type}",
                    f"Downloaded {current}/{total} MB ({progress:.1f}%)",
                    progress * 0.5  # Reserve 50% for download, 50% for loading
                )
        
        return progress_callback
    
    @classmethod
    @lru_cache(maxsize=1)
    def get_visual_model(cls, job_id: Optional[str] = None, db: Optional[Session] = None):
        """Load and cache visual sentiment analysis model."""
        if "visual_model" not in cls._models:
            # Check for local model first
            local_path = cls.get_local_model_path("visual")
            
            if local_path:
                cls.update_progress(job_id, db, "Loading Visual Model", "Loading local facial emotion detection model...", 10)
                print(f"Loading visual model from local path: {local_path}")
                
                cls.update_progress(job_id, db, "Loading Visual Model", "Loading local image processor...", 15)
                cls._processors["visual_processor"] = AutoImageProcessor.from_pretrained(local_path)
                
                cls.update_progress(job_id, db, "Loading Visual Model", "Loading local classification model...", 20)
                cls._models["visual_model"] = AutoModelForImageClassification.from_pretrained(local_path)
                
                cls.update_progress(job_id, db, "Visual Model Ready", "Local facial emotion detection model loaded successfully", 25)
                print("Visual model loaded from local files.")
            else:
                # Fallback to downloading from Hugging Face
                model_name = "dima806/facial_emotions_image_detection"
                
                cls.update_progress(job_id, db, "Loading Visual Model", "Downloading facial emotion detection model...", 10)
                print("Loading visual model from Hugging Face...")
                
                cls.update_progress(job_id, db, "Loading Visual Model", "Loading image processor...", 15)
                cls._processors["visual_processor"] = AutoImageProcessor.from_pretrained(model_name)
                
                cls.update_progress(job_id, db, "Loading Visual Model", "Loading classification model...", 20)
                cls._models["visual_model"] = AutoModelForImageClassification.from_pretrained(model_name)
                
                cls.update_progress(job_id, db, "Visual Model Ready", "Facial emotion detection model loaded successfully", 25)
                print("Visual model loaded from Hugging Face.")
                
        return cls._models["visual_model"], cls._processors["visual_processor"]
    
    @classmethod
    @lru_cache(maxsize=1)
    def get_text_model(cls, job_id: Optional[str] = None, db: Optional[Session] = None):
        """Load and cache text sentiment analysis model."""
        if "text_model" not in cls._models:
            # Check for local model first
            local_path = cls.get_local_model_path("text")
            
            if local_path:
                cls.update_progress(job_id, db, "Loading Text Model", "Loading local text sentiment analysis model...", 30)
                print(f"Loading text model from local path: {local_path}")
                
                cls._models["text_model"] = AutoModel.from_pretrained(local_path)
                
                cls.update_progress(job_id, db, "Text Model Ready", "Local text sentiment analysis model loaded successfully", 35)
                print("Text model loaded from local files.")
            else:
                # Fallback to downloading from Hugging Face
                model_name = "testnew21/text-model"
                
                cls.update_progress(job_id, db, "Loading Text Model", "Downloading text sentiment analysis model...", 30)
                print("Loading text model from Hugging Face...")
                
                cls._models["text_model"] = AutoModel.from_pretrained(model_name)
                
                cls.update_progress(job_id, db, "Text Model Ready", "Text sentiment analysis model loaded successfully", 35)
                print("Text model loaded from Hugging Face.")
                
        return cls._models["text_model"]
    
    @classmethod
    @lru_cache(maxsize=1)
    def get_audio_model(cls, job_id: Optional[str] = None, db: Optional[Session] = None):
        """Load and cache audio sentiment analysis model."""
        if "audio_model" not in cls._models:
            # Check for local model first
            local_path = cls.get_local_model_path("audio")
            
            if local_path:
                cls.update_progress(job_id, db, "Loading Audio Model", "Loading local audio sentiment analysis model...", 40)
                print(f"Loading audio model from local path: {local_path}")
                
                cls._models["audio_model"] = AutoModel.from_pretrained(local_path)
                
                cls.update_progress(job_id, db, "Audio Model Ready", "Local audio sentiment analysis model loaded successfully", 45)
                print("Audio model loaded from local files.")
            else:
                # Fallback to downloading from Hugging Face
                model_name = "testnew21/audio-model"
                
                cls.update_progress(job_id, db, "Loading Audio Model", "Downloading audio sentiment analysis model...", 40)
                print("Loading audio model from Hugging Face...")
                
                cls._models["audio_model"] = AutoModel.from_pretrained(model_name)
                
                cls.update_progress(job_id, db, "Audio Model Ready", "Audio sentiment analysis model loaded successfully", 45)
                print("Audio model loaded from Hugging Face.")
                
        return cls._models["audio_model"]
    
    @classmethod
    @lru_cache(maxsize=1)
    def get_transcription_model(cls, job_id: Optional[str] = None, db: Optional[Session] = None):
        """Load and cache Whisper transcription model using Hugging Face Transformers."""
        if "transcription_model" not in cls._models or "transcription_processor" not in cls._processors:
            # Check for local model first
            local_path = cls.get_local_model_path("whisper")
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if local_path:
                cls.update_progress(job_id, db, "Loading Whisper Model", 
                                  "Loading local Whisper speech-to-text model...", 50)
                print(f"Loading whisper model from local path: {local_path}")
                
                cls.update_progress(job_id, db, "Loading Whisper Model", 
                                  f"Loading local Whisper processor...", 55)
                
                # Load processor
                cls._processors["transcription_processor"] = WhisperProcessor.from_pretrained(local_path)
                
                cls.update_progress(job_id, db, "Loading Whisper Model", 
                                  f"Loading local Whisper model...", 70)
                
                # Load model
                cls._models["transcription_model"] = WhisperForConditionalGeneration.from_pretrained(local_path)
                
                # Move to device
                cls.update_progress(job_id, db, "Loading Whisper Model", 
                                  f"Loading local Whisper model on {device}...", 85)
                
                cls._models["transcription_model"] = cls._models["transcription_model"].to(device)
                
                cls.update_progress(job_id, db, "Whisper Model Ready", 
                                  f"Local speech-to-text model loaded successfully on {device}", 90)
                print(f"Whisper model loaded from local files on {device}.")
            else:
                # Fallback to downloading from Hugging Face
                model_name = "openai/whisper-base"
                
                cls.update_progress(job_id, db, "Loading Whisper Model", 
                                  "Downloading Whisper speech-to-text model (this may take a few minutes)...", 50)
                print("Loading whisper model from Hugging Face...")
                
                cls.update_progress(job_id, db, "Loading Whisper Model", 
                                  f"Downloading Whisper processor from {model_name}...", 55)
                
                # Load processor
                cls._processors["transcription_processor"] = WhisperProcessor.from_pretrained(model_name)
                
                cls.update_progress(job_id, db, "Loading Whisper Model", 
                                  f"Downloading Whisper model from {model_name}...", 70)
                
                # Load model
                cls._models["transcription_model"] = WhisperForConditionalGeneration.from_pretrained(model_name)
                
                # Move to device
                cls.update_progress(job_id, db, "Loading Whisper Model", 
                                  f"Loading Whisper model on {device}...", 85)
                
                cls._models["transcription_model"] = cls._models["transcription_model"].to(device)
                
                cls.update_progress(job_id, db, "Whisper Model Ready", 
                                  f"Speech-to-text model loaded successfully on {device}", 90)
                print(f"Whisper model loaded from Hugging Face on {device}.")
            
        return cls._models["transcription_model"], cls._processors["transcription_processor"]
    
    @classmethod
    def cleanup(cls):
        """Clear all loaded models from memory."""
        for key in list(cls._models.keys()):
            if key in cls._models:
                del cls._models[key]
        for key in list(cls._processors.keys()):
            if key in cls._processors:
                del cls._processors[key]
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Model cache cleared.")
    
    @classmethod
    def check_local_models_status(cls):
        """Check which models are available locally."""
        model_types = ["visual", "text", "audio", "whisper"]
        status = {}
        
        for model_type in model_types:
            local_path = cls.get_local_model_path(model_type)
            status[model_type] = {
                "available": local_path is not None,
                "path": str(local_path) if local_path else None
            }
        
        return status 