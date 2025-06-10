#!/usr/bin/env python3
"""
Simple Video Sentiment Analyzer - Direct Processing
No API, no background tasks, just direct video analysis using local models.
"""

import os
import sys
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
import json
from datetime import datetime

class SimpleVideoAnalyzer:
    def __init__(self):
        self.models_dir = Path("backend/models")
        self.models = {}
        self.processors = {}
        
    def load_models(self):
        """Load all models from local directories."""
        print("🔄 Loading AI models...")
        
        # Visual Model
        visual_path = self.models_dir / "visual_emotion_detection"
        if visual_path.exists():
            print("📸 Loading Visual Emotion Detection model...")
            self.processors["visual"] = AutoImageProcessor.from_pretrained(visual_path)
            self.models["visual"] = AutoModelForImageClassification.from_pretrained(visual_path)
            print("✅ Visual model loaded")
        else:
            print("❌ Visual model not found at", visual_path)
            return False
            
        # Text Model
        text_path = self.models_dir / "text_sentiment"
        if text_path.exists():
            print("📝 Loading Text Sentiment model...")
            self.models["text"] = AutoModel.from_pretrained(text_path)
            print("✅ Text model loaded")
        else:
            print("❌ Text model not found at", text_path)
            return False
            
        # Audio Model
        audio_path = self.models_dir / "audio_sentiment"
        if audio_path.exists():
            print("🔊 Loading Audio Sentiment model...")
            self.models["audio"] = AutoModel.from_pretrained(audio_path)
            print("✅ Audio model loaded")
        else:
            print("❌ Audio model not found at", audio_path)
            return False
            
        # Whisper Model
        whisper_path = self.models_dir / "whisper_base"
        if whisper_path.exists():
            print("🎤 Loading Whisper Speech-to-Text model...")
            self.processors["whisper"] = WhisperProcessor.from_pretrained(whisper_path)
            self.models["whisper"] = WhisperForConditionalGeneration.from_pretrained(whisper_path)
            
            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.models["whisper"] = self.models["whisper"].to(device)
            print(f"✅ Whisper model loaded on {device}")
        else:
            print("❌ Whisper model not found at", whisper_path)
            return False
            
        print("🎉 All models loaded successfully!")
        return True
    
    def extract_frames(self, video_path, max_frames=10):
        """Extract frames from video."""
        print("🎬 Extracting frames from video...")
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract frames evenly distributed throughout the video
        frame_indices = np.linspace(0, frame_count - 1, min(max_frames, frame_count), dtype=int)
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                print(f"  📷 Extracted frame {i+1}/{len(frame_indices)}")
        
        cap.release()
        print(f"✅ Extracted {len(frames)} frames")
        return frames
    
    def extract_audio(self, video_path):
        """Extract audio from video."""
        print("🔊 Extracting audio from video...")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            try:
                # Extract audio using moviepy
                video = mp.VideoFileClip(video_path)
                audio = video.audio
                audio.write_audiofile(temp_audio.name, verbose=False, logger=None)
                video.close()
                audio.close()
                
                print("✅ Audio extracted successfully")
                return temp_audio.name
            except Exception as e:
                print(f"❌ Error extracting audio: {str(e)}")
                return None
    
    def analyze_visual_emotions(self, frames):
        """Analyze emotions from video frames."""
        print("👁️ Analyzing visual emotions...")
        
        model = self.models["visual"]
        processor = self.processors["visual"]
        
        emotion_predictions = []
        
        for i, frame in enumerate(frames):
            try:
                # Process frame
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
                
                print(f"  🎭 Frame {i+1}: {predicted_label} ({confidence:.2f})")
                
            except Exception as e:
                print(f"  ❌ Error processing frame {i+1}: {str(e)}")
                continue
        
        # Calculate average sentiment
        if emotion_predictions:
            # Map emotions to sentiment scores (0=negative, 1=positive)
            emotion_to_sentiment = {
                "angry": 0.1, "disgust": 0.1, "fear": 0.2, "sad": 0.1,
                "happy": 0.9, "surprise": 0.7, "neutral": 0.5
            }
            
            sentiment_scores = []
            for pred in emotion_predictions:
                emotion = pred["emotion"].lower()
                score = emotion_to_sentiment.get(emotion, 0.5)
                sentiment_scores.append(score * pred["confidence"])
            
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.5
            dominant_emotion = max(emotion_predictions, key=lambda x: x["confidence"])["emotion"]
            
            print(f"✅ Visual analysis complete - Sentiment: {avg_sentiment:.2f}, Dominant: {dominant_emotion}")
            
            return {
                "sentiment_score": avg_sentiment,
                "dominant_emotion": dominant_emotion,
                "emotions": emotion_predictions
            }
        else:
            return {"sentiment_score": 0.5, "dominant_emotion": "neutral", "emotions": []}
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio to text."""
        print("🎤 Transcribing audio to text...")
        
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
            
            print(f"✅ Transcription: '{transcription}'")
            return transcription
            
        except Exception as e:
            print(f"❌ Error transcribing audio: {str(e)}")
            return ""
    
    def analyze_text_sentiment(self, text):
        """Analyze sentiment from text."""
        print("📝 Analyzing text sentiment...")
        
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
            sentiment_score = 0.5  # Neutral
        else:
            sentiment_score = positive_count / total
        
        dominant = "positive" if sentiment_score > 0.6 else "negative" if sentiment_score < 0.4 else "neutral"
        
        print(f"✅ Text sentiment: {sentiment_score:.2f} ({dominant})")
        
        return {
            "sentiment_score": sentiment_score,
            "dominant_emotion": dominant,
            "positive_words": positive_count,
            "negative_words": negative_count
        }
    
    def analyze_audio_sentiment(self, audio_path):
        """Analyze sentiment from audio features."""
        print("🔊 Analyzing audio sentiment...")
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Extract features
            energy = np.mean(librosa.feature.rms(y=y)[0])
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Normalize features
            norm_energy = min(1.0, energy * 10)
            norm_pitch = min(1.0, pitch / 500)
            norm_tempo = min(1.0, tempo / 180)
            
            # Combine for sentiment
            sentiment_score = (norm_energy + norm_pitch + norm_tempo) / 3
            dominant = "positive" if sentiment_score > 0.6 else "negative" if sentiment_score < 0.4 else "neutral"
            
            print(f"✅ Audio sentiment: {sentiment_score:.2f} ({dominant})")
            
            return {
                "sentiment_score": sentiment_score,
                "dominant_emotion": dominant,
                "features": {
                    "energy": energy,
                    "pitch": pitch,
                    "tempo": tempo
                }
            }
            
        except Exception as e:
            print(f"❌ Error analyzing audio: {str(e)}")
            return {"sentiment_score": 0.5, "dominant_emotion": "neutral"}
    
    def combine_results(self, visual_result, text_result, audio_result):
        """Combine all sentiment analysis results."""
        print("🔄 Combining results...")
        
        # Weights for different modalities
        weights = {"visual": 0.4, "text": 0.3, "audio": 0.3}
        
        combined_score = (
            visual_result["sentiment_score"] * weights["visual"] +
            text_result["sentiment_score"] * weights["text"] +
            audio_result["sentiment_score"] * weights["audio"]
        )
        
        # Calculate confidence based on agreement
        scores = [visual_result["sentiment_score"], text_result["sentiment_score"], audio_result["sentiment_score"]]
        confidence = 1 - (max(scores) - min(scores))
        
        overall_sentiment = "positive" if combined_score > 0.6 else "negative" if combined_score < 0.4 else "neutral"
        
        print(f"✅ Combined sentiment: {combined_score:.2f} ({overall_sentiment}) - Confidence: {confidence:.2f}")
        
        return {
            "combined_sentiment": combined_score,
            "overall_sentiment": overall_sentiment,
            "confidence": confidence,
            "breakdown": {
                "visual": visual_result,
                "text": text_result,
                "audio": audio_result
            }
        }
    
    def analyze_video(self, video_path):
        """Analyze a video file for sentiment."""
        print(f"\n🎬 Starting analysis of: {video_path}")
        print("=" * 60)
        
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return None
        
        # Load models
        if not self.load_models():
            print("❌ Failed to load models")
            return None
        
        try:
            # Extract frames
            frames = self.extract_frames(video_path)
            if not frames:
                print("❌ No frames extracted")
                return None
            
            # Extract audio
            audio_path = self.extract_audio(video_path)
            if not audio_path:
                print("❌ No audio extracted")
                return None
            
            # Analyze visual emotions
            visual_result = self.analyze_visual_emotions(frames)
            
            # Transcribe audio
            transcription = self.transcribe_audio(audio_path)
            
            # Analyze text sentiment
            text_result = self.analyze_text_sentiment(transcription)
            
            # Analyze audio sentiment
            audio_result = self.analyze_audio_sentiment(audio_path)
            
            # Combine results
            final_result = self.combine_results(visual_result, text_result, audio_result)
            
            # Add metadata
            final_result["metadata"] = {
                "video_file": video_path,
                "transcription": transcription,
                "analysis_time": datetime.now().isoformat(),
                "frames_analyzed": len(frames)
            }
            
            # Clean up temporary audio file
            try:
                os.unlink(audio_path)
            except:
                pass
            
            print("\n🎉 Analysis Complete!")
            print("=" * 60)
            
            return final_result
            
        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            return None

def main():
    """Main function to run video analysis."""
    if len(sys.argv) != 2:
        print("Usage: python simple_video_analyzer.py <video_file>")
        print("Example: python simple_video_analyzer.py my_video.mp4")
        return
    
    video_path = sys.argv[1]
    
    analyzer = SimpleVideoAnalyzer()
    result = analyzer.analyze_video(video_path)
    
    if result:
        # Save results to JSON file
        output_file = f"analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n📊 RESULTS SUMMARY")
        print("=" * 40)
        print(f"Overall Sentiment: {result['overall_sentiment'].upper()}")
        print(f"Sentiment Score: {result['combined_sentiment']:.2f}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Transcription: '{result['metadata']['transcription']}'")
        print(f"\nDetailed results saved to: {output_file}")
    else:
        print("❌ Analysis failed")

if __name__ == "__main__":
    main() 