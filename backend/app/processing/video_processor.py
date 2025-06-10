import os
import cv2
import numpy as np
from pathlib import Path
from moviepy.editor import VideoFileClip
import tempfile
from ..utils.file_handler import get_temp_dir


def extract_audio_from_video(video_path: str, job_id: str) -> str:
    """
    Extract audio from video file.
    
    Args:
        video_path: Path to video file
        job_id: Job ID for temporary directory
        
    Returns:
        str: Path to extracted audio file
    """
    # Get temp directory for this job
    temp_dir = get_temp_dir(job_id)
    
    # Output audio path
    audio_path = temp_dir / "audio.wav"
    
    # Load video
    video = VideoFileClip(video_path)
    
    # Extract audio
    video.audio.write_audiofile(str(audio_path), codec='pcm_s16le', fps=16000, nbytes=2, logger=None)
    
    return str(audio_path)


def extract_frames_from_video(video_path: str, job_id: str, frame_interval: int = 24) -> list[str]:
    """
    Extract frames from video file at regular intervals.
    
    Args:
        video_path: Path to video file
        job_id: Job ID for temporary directory
        frame_interval: Extract 1 frame every N frames
        
    Returns:
        list: List of paths to extracted frame images
    """
    # Get temp directory for this job
    temp_dir = get_temp_dir(job_id)
    frames_dir = temp_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate total frames to extract (limit to 100 frames maximum)
    max_frames = min(100, frame_count // frame_interval)
    
    frame_paths = []
    frame_idx = 0
    frames_extracted = 0
    
    while frame_idx < frame_count and frames_extracted < max_frames:
        # Set position to desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save frame
        output_path = frames_dir / f"frame_{frames_extracted:04d}.jpg"
        cv2.imwrite(str(output_path), frame)
        frame_paths.append(str(output_path))
        
        frames_extracted += 1
        frame_idx += frame_interval
    
    # Release video capture
    cap.release()
    
    return frame_paths


def get_video_metadata(video_path: str) -> dict:
    """
    Get metadata from video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        dict: Video metadata
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate duration
    duration = frame_count / fps if fps > 0 else 0
    
    # Release video capture
    cap.release()
    
    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
    } 