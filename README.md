# Multi-Modal Sentiment Analysis

A full-stack web application for analyzing sentiment in videos through visual, audio, and text modalities.

## Features
- Upload video files (MP4, AVI, MOV)
- Real-time processing status updates
- Sentiment analysis through three modalities:
  - Visual emotion detection
  - Audio sentiment analysis
  - Text sentiment analysis (via transcription)
- Combined sentiment score with confidence level
- Downloadable results

## Tech Stack
- Frontend: Next.js with TypeScript and Tailwind CSS
- Backend: FastAPI with Python 3.9+
- Models: 
  - Visual: dima806/facial_emotions_image_detection
  - Text: testnew21/text-model
  - Audio: testnew21/audio-model
  - Transcription: RedHatAI/whisper-large-v3-quantized.w4a16

## Setup Instructions
### 1. Clone this repository
### 2. Download the models:
   
  #### Step 1: Check Current Status
  ```
  cd backend
  python check_models.py
  ```
        
  #### Step 2: Download All Models (One-Time Setup)
  ```bash
  python download_models.py
  ```
  This will take 10-30 minutes depending on your internet speed, but you only need to do it once.

### 3. Set up the backend:
   ```
   cd backend
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```
### 4. Set up the frontend:
   ```
   cd frontend
   npm install
   npm run dev
   ```

Feel free to contribute and star the repository for latest updates.