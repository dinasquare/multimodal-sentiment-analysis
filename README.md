# Multi-Modal Sentiment Analysis 🎥🗣️📝

A full-stack web application for analyzing sentiment in videos through visual, audio, and text modalities. This application provides a holistic understanding of the emotional context of a video by combining visual emotion detection, audio sentiment analysis, and text sentiment analysis from transcriptions.

## About The Project

This project is designed to offer a comprehensive sentiment analysis of video content. Users can upload video files in various formats (MP4, AVI, MOV) and receive a detailed breakdown of the sentiment expressed through different modalities. The application processes the video to extract visual cues from facial expressions, analyze the emotional tone of the audio, and transcribe the speech to text for sentiment analysis. These individual analyses are then combined to produce a unified sentiment score with a confidence level, giving a more nuanced and accurate understanding of the video's emotional content.

The frontend is built with Next.js, providing a modern, responsive user interface, while the backend is powered by FastAPI, ensuring high performance and scalability. The sentiment analysis itself is performed by a suite of machine learning models, including facial emotion detection, audio sentiment analysis, and speech-to-text transcription.

### Features

-   **Video Upload**: Supports MP4, AVI, and MOV video file formats.
-   **Multi-Modal Analysis**:
    -   **Visual Sentiment**: Detects facial emotions from video frames.
    -   **Audio Sentiment**: Analyzes the sentiment from the audio track.
    -   **Text Sentiment**: Transcribes the audio to text and then analyzes the sentiment of the transcribed text.
-   **Combined Score**: Provides a combined sentiment score with a confidence level for a comprehensive analysis.
-   **Real-time Updates**: Users can view the status of the video processing in real-time.
-   **Downloadable Results**: The analysis results can be downloaded as a PDF file for offline viewing and sharing.

## Tech Stack

This project uses the following technologies:

-   **Frontend**: Next.js, TypeScript, Tailwind CSS, React, Chart.js
-   **Backend**: FastAPI, Python 3.9+
-   **Database**: SQLAlchemy
-   **ML Models**:
    -   **Visual**: `dima806/facial_emotions_image_detection`
    -   **Text**: `testnew21/text-model`
    -   **Audio**: `testnew21/audio-model`
    -   **Transcription**: `RedHatAI/whisper-large-v3-quantized.w4a16`

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

-   Python 3.9+
-   Node.js and npm
-   ffmpeg

### Installation

1.  **Clone the repository**:
    ```sh
    git clone [https://github.com/dinasquare/multimodal-sentiment-analysis.git](https://github.com/dinasquare/multimodal-sentiment-analysis.git)
    cd multimodal-sentiment-analysis
    ```

2.  **Download the models**:
    This is a one-time setup that will significantly speed up the processing time.

    -   First, check the status of the models:
        ```sh
        cd backend
        python check_models.py
        ```
    -   Then, download all the models:
        ```bash
        python download_models.py
        ```
        This might take some time depending on your internet connection.

3.  **Set up the backend**:
    ```sh
    cd backend
    pip install -r requirements.txt
    uvicorn app.main:app --reload
    ```

4.  **Set up the frontend**:
    ```sh
    cd frontend
    npm install
    npm run dev
    ```

## API Endpoints

The backend provides the following API endpoints:

| Endpoint                | Method | Description                                       |
| ----------------------- | ------ | ------------------------------------------------- |
| `/api/v1/upload`        | `POST` | Upload a video file for sentiment analysis.       |
| `/api/v1/status/{job_id}` | `GET`  | Check the status of a sentiment analysis job.     |
| `/api/v1/results/{job_id}`| `GET`  | Get the results of a completed sentiment analysis job. |
| `/api/v1/cleanup/{job_id}`| `DELETE`| Clean up files associated with a job.             |

For more details, refer to the `API_DOCS.md` file.

## How to Use

1.  Once the application is running, open your browser and navigate to `http://localhost:3000`.
2.  Upload a video file (MP4, AVI, or MOV).
3.  The application will start processing the video, and you can see the real-time status updates.
4.  Once the processing is complete, the results will be displayed on the screen, including the combined sentiment score, confidence level, and a breakdown of the visual, audio, and text sentiment analysis.
5.  You can download the results as a PDF file by clicking on the "Download PDF" button.

## License

Distributed under the MIT License. See `LICENSE` for more information.
