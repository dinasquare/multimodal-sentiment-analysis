import os
import torch
import numpy as np
import whisper
import cv2
from pathlib import Path
import librosa
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional
from sqlalchemy.orm import Session
from ..utils.model_loader import ModelLoader


def analyze_visual_sentiment(frame_paths: List[str], job_id: Optional[str] = None, db: Optional[Session] = None) -> Dict[str, Any]:
    """
    Analyze visual sentiment from video frames.
    
    Args:
        frame_paths: List of paths to video frames
        job_id: Optional job ID for progress tracking
        db: Optional database session for progress tracking
        
    Returns:
        dict: Visual sentiment analysis results
    """
    # Load model and processor with progress tracking
    model, processor = ModelLoader.get_visual_model(job_id, db)
    
    # Store emotion predictions
    emotion_predictions = []
    
    # Process each frame
    for i, frame_path in enumerate(frame_paths):
        try:
            # Update progress for frame processing
            if job_id and db:
                progress = (i / len(frame_paths)) * 100
                ModelLoader.update_progress(job_id, db, "Analyzing Visual Emotions", 
                                          f"Processing frame {i+1}/{len(frame_paths)} ({progress:.1f}%)")
            
            # Open image
            image = Image.open(frame_path)
            
            # Process image for model input
            inputs = processor(images=image, return_tensors="pt")
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
            # Get predicted class and probabilities
            predicted_class_idx = logits.argmax(-1).item()
            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
            
            # Get class labels
            predicted_label = model.config.id2label[predicted_class_idx]
            
            # Store prediction for this frame
            frame_prediction = {
                "label": predicted_label,
                "score": probabilities[predicted_class_idx].item(),
                "all_emotions": {
                    model.config.id2label[i]: prob.item()
                    for i, prob in enumerate(probabilities)
                }
            }
            
            emotion_predictions.append(frame_prediction)
            
        except Exception as e:
            print(f"Error processing frame {frame_path}: {str(e)}")
            continue
    
    # If no frames were successfully processed
    if not emotion_predictions:
        return {
            "sentiment_score": 0.5,  # Neutral score
            "dominant_emotion": "neutral",
            "emotions": []
        }
    
    # Calculate average emotion scores across all frames
    all_emotions = {}
    for pred in emotion_predictions:
        for emotion, score in pred["all_emotions"].items():
            if emotion not in all_emotions:
                all_emotions[emotion] = []
            all_emotions[emotion].append(score)
    
    avg_emotions = {
        emotion: sum(scores) / len(scores)
        for emotion, scores in all_emotions.items()
    }
    
    # Get dominant emotion and its score
    dominant_emotion = max(avg_emotions.items(), key=lambda x: x[1])
    
    # Convert emotions to list format for response
    emotions_list = [
        {"label": emotion, "score": score}
        for emotion, score in avg_emotions.items()
    ]
    
    # Map emotions to sentiment score (0-1 scale)
    # Mapping based on valence of emotions
    emotion_valence = {
        "happy": 0.9,
        "joy": 0.9,
        "surprise": 0.7,
        "neutral": 0.5,
        "sad": 0.3,
        "fear": 0.2,
        "disgust": 0.2,
        "anger": 0.1,
        "angry": 0.1
    }
    
    # Calculate sentiment score based on weighted emotion valence
    sentiment_score = 0.5  # Default to neutral
    total_weight = 0
    
    for emotion, score in avg_emotions.items():
        # Get valence for this emotion (default to 0.5 if unknown)
        valence = emotion_valence.get(emotion.lower(), 0.5)
        sentiment_score += valence * score
        total_weight += score
    
    if total_weight > 0:
        sentiment_score /= total_weight
    
    return {
        "sentiment_score": sentiment_score,
        "dominant_emotion": dominant_emotion[0],
        "emotions": emotions_list
    }


def analyze_text_sentiment(text: str, job_id: Optional[str] = None, db: Optional[Session] = None) -> Dict[str, Any]:
    """
    Analyze sentiment from text.
    
    Args:
        text: Text to analyze
        job_id: Optional job ID for progress tracking
        db: Optional database session for progress tracking
        
    Returns:
        dict: Text sentiment analysis results
    """
    if job_id and db:
        ModelLoader.update_progress(job_id, db, "Analyzing Text Sentiment", "Processing transcribed text...")
    
    # For now, we'll use a placeholder implementation
    # In a real implementation, you would use the text model to perform sentiment analysis
    
    # Placeholder implementation
    # Count positive and negative words
    positive_words = ["good", "great", "happy", "positive", "excellent", "wonderful", "love", "joy"]
    negative_words = ["bad", "terrible", "sad", "negative", "awful", "hate", "anger", "fear"]
    
    text = text.lower()
    words = text.split()
    
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    total = positive_count + negative_count
    if total == 0:
        sentiment_score = 0.5  # Neutral
    else:
        sentiment_score = positive_count / (positive_count + negative_count)
    
    # Create mock emotions list
    emotions = [
        {"label": "positive", "score": sentiment_score},
        {"label": "negative", "score": 1 - sentiment_score},
        {"label": "neutral", "score": 0.5 * (1 - abs(sentiment_score - 0.5) * 2)}
    ]
    
    return {
        "sentiment_score": sentiment_score,
        "dominant_emotion": "positive" if sentiment_score > 0.6 else "negative" if sentiment_score < 0.4 else "neutral",
        "emotions": emotions
    }


def analyze_audio_sentiment(audio_path: str, job_id: Optional[str] = None, db: Optional[Session] = None) -> Dict[str, Any]:
    """
    Analyze sentiment from audio.
    
    Args:
        audio_path: Path to audio file
        job_id: Optional job ID for progress tracking
        db: Optional database session for progress tracking
        
    Returns:
        dict: Audio sentiment analysis results
    """
    if job_id and db:
        ModelLoader.update_progress(job_id, db, "Analyzing Audio Sentiment", "Extracting audio features...")
    
    # For now, we'll use a placeholder implementation
    # In a real implementation, you would use the audio model to perform sentiment analysis
    
    # Placeholder implementation
    # Extract audio features and make a simple heuristic
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Get audio features
        # Energy - higher energy might indicate excitement/happiness
        energy = np.mean(librosa.feature.rms(y=y)[0])
        
        # Pitch (fundamental frequency using harmonic product spectrum)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Simple heuristics - higher energy, pitch and tempo might indicate more positive emotions
        # Normalize values between 0 and 1
        norm_energy = min(1.0, energy * 10)  # Scale energy (typically small values)
        norm_pitch = min(1.0, pitch / 500)   # Scale pitch (typical human range is 80-400 Hz)
        norm_tempo = min(1.0, tempo / 180)   # Scale tempo (typical range 60-180 BPM)
        
        # Combine features for sentiment
        sentiment_score = (norm_energy + norm_pitch + norm_tempo) / 3
        
        # Create mock emotions list
        emotions = [
            {"label": "happy", "score": sentiment_score},
            {"label": "sad", "score": 1 - sentiment_score},
            {"label": "neutral", "score": 0.5 * (1 - abs(sentiment_score - 0.5) * 2)}
        ]
        
        return {
            "sentiment_score": sentiment_score,
            "dominant_emotion": "happy" if sentiment_score > 0.6 else "sad" if sentiment_score < 0.4 else "neutral",
            "emotions": emotions
        }
    
    except Exception as e:
        print(f"Error analyzing audio: {str(e)}")
        return {
            "sentiment_score": 0.5,  # Neutral score as fallback
            "dominant_emotion": "neutral",
            "emotions": [{"label": "neutral", "score": 1.0}]
        }


def transcribe_audio(audio_path: str, job_id: Optional[str] = None, db: Optional[Session] = None) -> str:
    """
    Transcribe speech from audio file using Hugging Face Transformers Whisper.
    
    Args:
        audio_path: Path to audio file
        job_id: Optional job ID for progress tracking
        db: Optional database session for progress tracking
        
    Returns:
        str: Transcribed text
    """
    import librosa
    import torch
    
    if job_id and db:
        ModelLoader.update_progress(job_id, db, "Transcribing Audio", "Converting speech to text...", 75)
    
    # Load model and processor with progress tracking
    model, processor = ModelLoader.get_transcription_model(job_id, db)
    
    if job_id and db:
        ModelLoader.update_progress(job_id, db, "Transcribing Audio", "Loading audio file...", 80)
    
    try:
        # Load audio file
        audio, sample_rate = librosa.load(audio_path, sr=16000)  # Whisper expects 16kHz
        
        if job_id and db:
            ModelLoader.update_progress(job_id, db, "Transcribing Audio", "Processing audio with Whisper...", 85)
        
        # Process audio with the processor
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        
        # Move inputs to the same device as the model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        if job_id and db:
            ModelLoader.update_progress(job_id, db, "Transcribing Audio", "Generating transcription...", 90)
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(**inputs)
        
        # Decode the transcription
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        if job_id and db:
            ModelLoader.update_progress(job_id, db, "Transcription Complete", 
                                      f"Transcribed text: {transcription[:100]}...", 95)
        
        return transcription
        
    except Exception as e:
        error_msg = f"Error transcribing audio: {str(e)}"
        print(error_msg)
        if job_id and db:
            ModelLoader.update_progress(job_id, db, "Transcription Error", error_msg, 95)
        return ""  # Return empty string on error


def fusion_algorithm(visual_result: Dict[str, Any], audio_result: Dict[str, Any], text_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine sentiment analysis results from multiple modalities.
    
    Args:
        visual_result: Visual sentiment analysis results
        audio_result: Audio sentiment analysis results
        text_result: Text sentiment analysis results
        
    Returns:
        dict: Combined sentiment analysis results
    """
    # Extract sentiment scores
    visual_sentiment = visual_result["sentiment_score"]
    audio_sentiment = audio_result["sentiment_score"]
    text_sentiment = text_result["sentiment_score"]
    
    # Define weights based on model reliability
    weights = {'visual': 0.35, 'audio': 0.35, 'text': 0.30}
    
    # Calculate weighted average
    combined_score = (
        visual_sentiment * weights['visual'] +
        audio_sentiment * weights['audio'] + 
        text_sentiment * weights['text']
    )
    
    # Calculate confidence based on agreement between modalities
    scores = [visual_sentiment, audio_sentiment, text_sentiment]
    confidence = 1 - (max(scores) - min(scores))
    
    return {
        'combined_sentiment': combined_score,
        'confidence': confidence,
        'breakdown': {
            'visual': visual_sentiment,
            'audio': audio_sentiment, 
            'text': text_sentiment,
            'visual_emotions': visual_result["emotions"],
            'audio_emotions': audio_result["emotions"],
            'text_emotions': text_result["emotions"]
        }
    } 