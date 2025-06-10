// Job status enum
export enum JobStatus {
  PENDING = "pending",
  PROCESSING = "processing",
  COMPLETED = "completed",
  FAILED = "failed",
}

// Job status response from API
export interface JobStatusResponse {
  id: string;
  status: JobStatus;
  error_message?: string;
  progress_percentage?: number;
  current_step?: string;
  step_details?: string;
  total_steps?: number;
}

// Model download info
export interface ModelDownloadInfo {
  modelName: string;
  modelType: string;
  downloadProgress: number;
  isDownloading: boolean;
}

// Emotion result from API
export interface EmotionResult {
  label: string;
  score: number;
}

// Sentiment result from API
export interface SentimentResult {
  visual_sentiment?: number;
  audio_sentiment?: number;
  text_sentiment?: number;
  combined_sentiment: number;
  confidence: number;
  visual_emotions?: EmotionResult[];
  audio_emotions?: EmotionResult[];
  text_emotions?: EmotionResult[];
}

// Job result from API
export interface JobResult {
  id: string;
  original_filename: string;
  created_at: string;
  updated_at: string;
  status: JobStatus;
  results?: SentimentResult;
  error_message?: string;
  progress_percentage?: number;
  current_step?: string;
  step_details?: string;
}

// Upload state for tracking upload progress
export interface UploadState {
  uploading: boolean;
  progress: number;
  jobId?: string;
  error?: string;
}

// Processing state for tracking job processing
export interface ProcessingState {
  processing: boolean;
  status: JobStatus;
  error?: string;
  progress_percentage?: number;
  current_step?: string;
  step_details?: string;
  currentModel?: ModelDownloadInfo;
}

// Combined app state
export interface AppState {
  uploadState: UploadState;
  processingState: ProcessingState;
  jobResult?: JobResult;
} 