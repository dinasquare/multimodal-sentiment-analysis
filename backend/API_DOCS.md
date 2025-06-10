# API Documentation

This document describes the endpoints available in the Multi-Modal Sentiment Analysis API.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API does not require authentication. For production use, consider implementing an authentication mechanism.

## Endpoints

### Upload Video

Upload a video file for sentiment analysis.

**URL**: `/upload`

**Method**: `POST`

**Content-Type**: `multipart/form-data`

**Parameters**:
- `file`: The video file to upload (MP4, AVI, MOV, max 100MB)

**Response**:
```json
{
  "id": "string",
  "status": "pending",
  "error_message": null
}
```

**Status Codes**:
- `200 OK`: Upload successful
- `400 Bad Request`: Invalid file type or size
- `500 Internal Server Error`: Server error

### Check Job Status

Check the status of a sentiment analysis job.

**URL**: `/status/{job_id}`

**Method**: `GET`

**Parameters**:
- `job_id`: The ID of the job to check (path parameter)

**Response**:
```json
{
  "id": "string",
  "status": "pending|processing|completed|failed",
  "error_message": null
}
```

**Status Codes**:
- `200 OK`: Status retrieved successfully
- `404 Not Found`: Job not found
- `500 Internal Server Error`: Server error

### Get Job Results

Get the results of a completed sentiment analysis job.

**URL**: `/results/{job_id}`

**Method**: `GET`

**Parameters**:
- `job_id`: The ID of the job to get results for (path parameter)

**Response**:
```json
{
  "id": "string",
  "original_filename": "string",
  "created_at": "string",
  "updated_at": "string",
  "status": "completed",
  "results": {
    "visual_sentiment": 0.75,
    "audio_sentiment": 0.65,
    "text_sentiment": 0.70,
    "combined_sentiment": 0.70,
    "confidence": 0.90,
    "visual_emotions": [
      {
        "label": "happy",
        "score": 0.85
      },
      {
        "label": "neutral",
        "score": 0.10
      },
      {
        "label": "sad",
        "score": 0.05
      }
    ],
    "audio_emotions": [
      {
        "label": "happy",
        "score": 0.75
      },
      {
        "label": "neutral",
        "score": 0.20
      },
      {
        "label": "sad",
        "score": 0.05
      }
    ],
    "text_emotions": [
      {
        "label": "positive",
        "score": 0.70
      },
      {
        "label": "neutral",
        "score": 0.25
      },
      {
        "label": "negative",
        "score": 0.05
      }
    ]
  },
  "error_message": null
}
```

**Status Codes**:
- `200 OK`: Results retrieved successfully
- `202 Accepted`: Job is still processing
- `404 Not Found`: Job not found
- `500 Internal Server Error`: Job failed or server error

### Clean Up Job Files

Clean up files associated with a job.

**URL**: `/cleanup/{job_id}`

**Method**: `DELETE`

**Parameters**:
- `job_id`: The ID of the job to clean up (path parameter)

**Response**:
```json
{
  "message": "Files for job {job_id} have been cleaned up"
}
```

**Status Codes**:
- `200 OK`: Cleanup successful
- `404 Not Found`: Job not found
- `500 Internal Server Error`: Server error

## Error Responses

All endpoints may return the following error response:

```json
{
  "detail": "Error message"
}
```

## Rate Limiting

Currently, the API does not implement rate limiting. For production use, consider adding rate limiting to prevent abuse. 