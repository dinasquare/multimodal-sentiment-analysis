#!/usr/bin/env python3
"""
Simplified Video Processor - Direct model usage without complexity
"""

import os
import cv2
import torch
import librosa
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification, 
    AutoModel,
    WhisperProcessor,
    WhisperForConditionalGeneration
)
import moviepy.editor as mp
import tempfile
from typing import Dict, Any, Optional

class SimpleProcessor:
    """Simple, direct video processor using local models."""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models = {}
        self.processors = {}
        self.models_loaded = False
    
    def load_models(self):
        """Load all models from local directories once."""
        if self.models_loaded:
            return True
            
        print("Loading models from local files...")
        
        try:
            # Visual Model
            visual_path = self.models_dir / "visual_emotion_detection"
            self.processors["visual"] = AutoImageProcessor.from_pretrained(visual_path)
            self.models["visual"] = AutoModelForImageClassification.from_pretrained(visual_path)
            
            # Text Model (not actually used in current implementation, but loaded for consistency)
            text_path = self.models_dir / "text_sentiment"
            self.models["text"] = AutoModel.from_pretrained(text_path)
            
            # Audio Model (not actually used in current implementation, but loaded for consistency)
            audio_path = self.models_dir / "audio_sentiment"
            self.models["audio"] = AutoModel.from_pretrained(audio_path)
            
            # Whisper Model
            whisper_path = self.models_dir / "whisper_base"
            self.processors["whisper"] = WhisperProcessor.from_pretrained(whisper_path)
            self.models["whisper"] = WhisperForConditionalGeneration.from_pretrained(whisper_path)
            
            # Move Whisper to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.models["whisper"] = self.models["whisper"].to(device)
            
            self.models_loaded = True
            print("✅ All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            return False
    
    def extract_frames(self, video_path: str, max_frames: int = 10):
        """Extract frames from video."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count == 0:
            cap.release()
            return frames
        
        # Extract frames evenly distributed throughout the video
        frame_indices = np.linspace(0, frame_count - 1, min(max_frames, frame_count), dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
        return frames
    
    def extract_audio(self, video_path: str) -> Optional[str]:
        """Extract audio from video."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                video = mp.VideoFileClip(video_path)
                if video.audio is not None:
                    audio = video.audio
                    audio.write_audiofile(temp_audio.name, verbose=False, logger=None)
                    video.close()
                    audio.close()
                    return temp_audio.name
                else:
                    video.close()
                    return None
        except Exception as e:
            print(f"Error extracting audio: {str(e)}")
            return None
    
    def analyze_visual_emotions(self, frames):
        """Analyze emotions from video frames."""
        if not frames:
            return {"sentiment_score": 0.5, "dominant_emotion": "neutral", "emotions": []}
        
        model = self.models["visual"]
        processor = self.processors["visual"]
        emotion_predictions = []
        
        for frame in frames:
            try:
                inputs = processor(images=frame, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                predicted_class_idx = logits.argmax(-1).item()
                probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
                
                predicted_label = model.config.id2label[predicted_class_idx]
                confidence = probabilities[predicted_class_idx].item()
                
                emotion_predictions.append({
                    "emotion": predicted_label,
                    "confidence": confidence
                })
                
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue
        
        if emotion_predictions:
            # Map emotions to sentiment scores
            emotion_to_sentiment = {
                "angry": 0.1, "disgust": 0.1, "fear": 0.2, "sad": 0.1,
                "happy": 0.9, "surprise": 0.7, "neutral": 0.5
            }
            
            sentiment_scores = []
            for pred in emotion_predictions:
                emotion = pred["emotion"].lower()
                score = emotion_to_sentiment.get(emotion, 0.5)
                sentiment_scores.append(score * pred["confidence"])
            
            avg_sentiment = np.mean(sentiment_scores)
            dominant_emotion = max(emotion_predictions, key=lambda x: x["confidence"])["emotion"]
            
            return {
                "sentiment_score": avg_sentiment,
                "dominant_emotion": dominant_emotion,
                "emotions": emotion_predictions
            }
        else:
            return {"sentiment_score": 0.5, "dominant_emotion": "neutral", "emotions": []}
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio to text."""
        if not audio_path or not os.path.exists(audio_path):
            return ""
        
        try:
            # Load audio
            audio, sample_rate = librosa.load(audio_path, sr=16000)
            
            # Process with Whisper
            processor = self.processors["whisper"]
            model = self.models["whisper"]
            
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            
            # Move inputs to same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                predicted_ids = model.generate(**inputs)
            
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription
            
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            return ""
    
    def analyze_text_sentiment(self, text: str):
        """Simple text sentiment analysis."""
        if not text.strip():
            return {"sentiment_score": 0.5, "dominant_emotion": "neutral"}
        
        # Simple keyword-based sentiment analysis
        positive_words = ["good", "great", "happy", "positive", "excellent", "wonderful", "love", "joy", "amazing", "fantastic"]
        negative_words = ["bad", "terrible", "sad", "negative", "awful", "hate", "anger", "fear", "horrible", "disgusting"]
        
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total = positive_count + negative_count
        if total == 0:
            sentiment_score = 0.5
        else:
            sentiment_score = positive_count / total
        
        dominant = "positive" if sentiment_score > 0.6 else "negative" if sentiment_score < 0.4 else "neutral"
        
        return {
            "sentiment_score": sentiment_score,
            "dominant_emotion": dominant
        }
    
    def analyze_audio_sentiment(self, audio_path: str):
        """Simple audio sentiment analysis."""
        if not audio_path or not os.path.exists(audio_path):
            return {"sentiment_score": 0.5, "dominant_emotion": "neutral"}
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Extract basic features
            energy = np.mean(librosa.feature.rms(y=y)[0])
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Simple normalization and combination
            norm_energy = min(1.0, energy * 10)
            norm_pitch = min(1.0, pitch / 500)
            norm_tempo = min(1.0, tempo / 180)
            
            sentiment_score = (norm_energy + norm_pitch + norm_tempo) / 3
            dominant = "positive" if sentiment_score > 0.6 else "negative" if sentiment_score < 0.4 else "neutral"
            
            return {
                "sentiment_score": sentiment_score,
                "dominant_emotion": dominant
            }
            
        except Exception as e:
            print(f"Error analyzing audio: {str(e)}")
            return {"sentiment_score": 0.5, "dominant_emotion": "neutral"}
    
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Process video and return results directly."""
        print(f"Processing video: {video_path}")
        
        # Load models if not already loaded
        if not self.load_models():
            return {"error": "Failed to load models"}
        
        try:
            # Extract frames and audio
            frames = self.extract_frames(video_path)
            audio_path = self.extract_audio(video_path)
            
            # Analyze visual emotions
            visual_result = self.analyze_visual_emotions(frames)
            
            # Transcribe audio
            transcription = self.transcribe_audio(audio_path) if audio_path else ""
            
            # Analyze text sentiment
            text_result = self.analyze_text_sentiment(transcription)
            
            # Analyze audio sentiment
            audio_result = self.analyze_audio_sentiment(audio_path)
            
            # Combine results
            weights = {"visual": 0.4, "text": 0.3, "audio": 0.3}
            combined_score = (
                visual_result["sentiment_score"] * weights["visual"] +
                text_result["sentiment_score"] * weights["text"] +
                audio_result["sentiment_score"] * weights["audio"]
            )
            
            # Calculate confidence
            scores = [visual_result["sentiment_score"], text_result["sentiment_score"], audio_result["sentiment_score"]]
            confidence = 1 - (max(scores) - min(scores))
            
            # Clean up temporary audio file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except:
                    pass
            
            print("✅ Video processing complete!")
            
            return {
                "visual_sentiment": visual_result["sentiment_score"],
                "audio_sentiment": audio_result["sentiment_score"],
                "text_sentiment": text_result["sentiment_score"],
                "combined_sentiment": combined_score,
                "confidence": confidence,
                "transcription": transcription,
                "visual_emotions": visual_result.get("emotions", []),
                "dominant_visual_emotion": visual_result["dominant_emotion"]
            }
            
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return {"error": f"Processing failed: {str(e)}"}

# Global processor instance
_processor = None

def get_processor():
    """Get the global processor instance."""
    global _processor
    if _processor is None:
        _processor = SimpleProcessor()
    return _processor 