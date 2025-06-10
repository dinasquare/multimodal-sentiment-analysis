#!/usr/bin/env python3
"""
Script to pre-download all AI models for the Multi-Modal Sentiment Analysis application.
This will download all model files locally to avoid download delays during processing.
"""

import os
import sys
from pathlib import Path
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification, 
    AutoModel,
    WhisperProcessor,
    WhisperForConditionalGeneration
)
import torch
from huggingface_hub import snapshot_download
import time

# Create models directory
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def download_model_with_progress(model_name: str, model_type: str, local_dir: Path):
    """Download a model with progress tracking."""
    print(f"\n{'='*60}")
    print(f"📥 Downloading {model_type}")
    print(f"Model: {model_name}")
    print(f"Local directory: {local_dir}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Download all model files using snapshot_download
        print("🔄 Downloading all model files...")
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Use actual files, not symlinks
            resume_download=True,  # Resume if interrupted
        )
        
        elapsed_time = time.time() - start_time
        print(f"✅ {model_type} downloaded successfully!")
        print(f"⏱️  Download time: {elapsed_time:.1f} seconds")
        print(f"📁 Files saved to: {local_dir}")
        
        # List downloaded files
        files = list(local_dir.rglob("*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        print(f"📊 Total files: {len([f for f in files if f.is_file()])}")
        print(f"💾 Total size: {total_size / (1024*1024):.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {model_type}: {str(e)}")
        return False

def test_model_loading(model_name: str, model_type: str, local_dir: Path):
    """Test loading the downloaded model."""
    print(f"\n🧪 Testing {model_type} loading...")
    
    try:
        if model_type == "Visual Emotion Detection":
            processor = AutoImageProcessor.from_pretrained(local_dir)
            model = AutoModelForImageClassification.from_pretrained(local_dir)
            print(f"✅ Visual model loaded successfully from {local_dir}")
            
        elif model_type == "Text Sentiment Analysis":
            model = AutoModel.from_pretrained(local_dir)
            print(f"✅ Text model loaded successfully from {local_dir}")
            
        elif model_type == "Audio Sentiment Analysis":
            model = AutoModel.from_pretrained(local_dir)
            print(f"✅ Audio model loaded successfully from {local_dir}")
            
        elif model_type == "Speech-to-Text (Whisper)":
            processor = WhisperProcessor.from_pretrained(local_dir)
            model = WhisperForConditionalGeneration.from_pretrained(local_dir)
            print(f"✅ Whisper model loaded successfully from {local_dir}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading {model_type}: {str(e)}")
        return False

def main():
    """Download all models for the sentiment analysis application."""
    print("🚀 Multi-Modal Sentiment Analysis - Model Downloader")
    print("=" * 60)
    print("This script will download all required AI models locally.")
    print("This is a one-time setup that will significantly speed up processing.")
    print("=" * 60)
    
    # Models to download
    models = [
        {
            "name": "dima806/facial_emotions_image_detection",
            "type": "Visual Emotion Detection",
            "local_dir": MODELS_DIR / "visual_emotion_detection"
        },
        {
            "name": "testnew21/text-model",
            "type": "Text Sentiment Analysis", 
            "local_dir": MODELS_DIR / "text_sentiment"
        },
        {
            "name": "testnew21/audio-model",
            "type": "Audio Sentiment Analysis",
            "local_dir": MODELS_DIR / "audio_sentiment"
        },
        {
            "name": "openai/whisper-base",
            "type": "Speech-to-Text (Whisper)",
            "local_dir": MODELS_DIR / "whisper_base"
        }
    ]
    
    print(f"📁 Models will be saved to: {MODELS_DIR.absolute()}")
    print(f"🔢 Total models to download: {len(models)}")
    
    # Ask for confirmation
    response = input("\n🤔 Do you want to proceed with downloading all models? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Download cancelled.")
        return
    
    # Download each model
    successful_downloads = 0
    total_start_time = time.time()
    
    for i, model_info in enumerate(models, 1):
        print(f"\n📦 [{i}/{len(models)}] Processing {model_info['type']}")
        
        # Create local directory
        model_info["local_dir"].mkdir(parents=True, exist_ok=True)
        
        # Check if model already exists
        if (model_info["local_dir"] / "config.json").exists():
            print(f"⚠️  Model already exists at {model_info['local_dir']}")
            response = input("🤔 Do you want to re-download? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("⏭️  Skipping download, testing existing model...")
                if test_model_loading(model_info["name"], model_info["type"], model_info["local_dir"]):
                    successful_downloads += 1
                continue
        
        # Download model
        if download_model_with_progress(model_info["name"], model_info["type"], model_info["local_dir"]):
            # Test loading
            if test_model_loading(model_info["name"], model_info["type"], model_info["local_dir"]):
                successful_downloads += 1
    
    # Summary
    total_elapsed_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print("📊 DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully downloaded: {successful_downloads}/{len(models)} models")
    print(f"⏱️  Total time: {total_elapsed_time:.1f} seconds")
    print(f"📁 Models location: {MODELS_DIR.absolute()}")
    
    if successful_downloads == len(models):
        print("\n🎉 All models downloaded successfully!")
        print("🚀 You can now run the application with much faster model loading!")
        print("\n📝 Next steps:")
        print("1. Update the model loader to use local models")
        print("2. Restart the backend server")
        print("3. Enjoy faster processing!")
    else:
        print(f"\n⚠️  {len(models) - successful_downloads} models failed to download.")
        print("❗ Please check your internet connection and try again.")
    
    # Calculate total disk usage
    try:
        total_size = sum(
            f.stat().st_size 
            for model_dir in [m["local_dir"] for m in models if m["local_dir"].exists()]
            for f in model_dir.rglob("*") if f.is_file()
        )
        print(f"💾 Total disk usage: {total_size / (1024*1024*1024):.2f} GB")
    except Exception as e:
        print(f"⚠️  Could not calculate disk usage: {str(e)}")

if __name__ == "__main__":
    main() 