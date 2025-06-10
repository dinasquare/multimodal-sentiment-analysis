#!/usr/bin/env python3
"""
Script to check the status of local AI models.
"""

from pathlib import Path
from app.utils.model_loader import ModelLoader

def main():
    """Check local models status."""
    print("🔍 Multi-Modal Sentiment Analysis - Model Status Check")
    print("=" * 60)
    
    # Check local models status
    status = ModelLoader.check_local_models_status()
    
    model_names = {
        "visual": "Visual Emotion Detection (dima806/facial_emotions_image_detection)",
        "text": "Text Sentiment Analysis (testnew21/text-model)",
        "audio": "Audio Sentiment Analysis (testnew21/audio-model)",
        "whisper": "Speech-to-Text Whisper (openai/whisper-base)"
    }
    
    local_count = 0
    total_count = len(status)
    
    for model_type, info in status.items():
        model_name = model_names[model_type]
        if info["available"]:
            print(f"✅ {model_name}")
            print(f"   📁 Path: {info['path']}")
            local_count += 1
        else:
            print(f"❌ {model_name}")
            print(f"   📥 Will be downloaded from Hugging Face on first use")
        print()
    
    print("=" * 60)
    print(f"📊 Summary: {local_count}/{total_count} models available locally")
    
    if local_count == total_count:
        print("🎉 All models are available locally! Processing will be much faster.")
    elif local_count > 0:
        print(f"⚡ {local_count} models are local, {total_count - local_count} will be downloaded on first use.")
    else:
        print("📥 No local models found. All models will be downloaded on first use.")
        print("💡 Run 'python download_models.py' to download all models locally.")
    
    print("\n💡 To download missing models locally, run:")
    print("   python download_models.py")

if __name__ == "__main__":
    main() 