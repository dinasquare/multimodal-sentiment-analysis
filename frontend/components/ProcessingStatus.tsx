import React, { useEffect, useState } from 'react';
import { api } from '../utils/apiClient';
import { JobStatus, JobStatusResponse, ModelDownloadInfo } from '../utils/types';

interface ProcessingStatusProps {
  jobId: string;
  onProcessingComplete: () => void;
  onProcessingError: (error: string) => void;
}

const ProcessingStatus: React.FC<ProcessingStatusProps> = ({
  jobId,
  onProcessingComplete,
  onProcessingError,
}) => {
  const [jobStatus, setJobStatus] = useState<JobStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pollingCount, setPollingCount] = useState(0);
  const [currentModel, setCurrentModel] = useState<ModelDownloadInfo | null>(null);

  useEffect(() => {
    let isMounted = true;
    let pollingInterval: NodeJS.Timeout;

    const checkStatus = async () => {
      try {
        const response = await api.getJobStatus(jobId);
        
        if (!isMounted) return;
        
        setJobStatus(response);
        setPollingCount(prev => prev + 1);
        
        // Parse model download information from step details
        if (response.step_details) {
          const modelInfo = parseModelDownloadInfo(response.current_step, response.step_details, response.progress_percentage);
          setCurrentModel(modelInfo);
        }
        
        if (response.status === JobStatus.COMPLETED) {
          clearInterval(pollingInterval);
          onProcessingComplete();
        } else if (response.status === JobStatus.FAILED) {
          clearInterval(pollingInterval);
          const errorMsg = response.error_message || 'Processing failed';
          setError(errorMsg);
          onProcessingError(errorMsg);
        }
      } catch (error) {
        if (!isMounted) return;
        
        console.error('Error checking job status:', error);
        const errorMsg = error instanceof Error ? error.message : 'Failed to check processing status';
        setError(errorMsg);
        clearInterval(pollingInterval);
        onProcessingError(errorMsg);
      }
    };

    // Check status immediately
    checkStatus();
    
    // Then poll every 1.5 seconds for more responsive updates during downloads
    pollingInterval = setInterval(checkStatus, 1500);
    
    return () => {
      isMounted = false;
      clearInterval(pollingInterval);
    };
  }, [jobId, onProcessingComplete, onProcessingError]);

  // Parse model download information from step details
  const parseModelDownloadInfo = (currentStep?: string, stepDetails?: string, progress?: number): ModelDownloadInfo | null => {
    if (!currentStep || !stepDetails) return null;
    
    const isDownloading = stepDetails.includes('Downloading') || stepDetails.includes('Downloaded');
    if (!isDownloading) return null;
    
    let modelName = '';
    let modelType = '';
    let downloadProgress = progress || 0;
    
    // Extract model information from step details
    if (stepDetails.includes('facial emotion detection')) {
      modelName = 'dima806/facial_emotions_image_detection';
      modelType = 'Visual Emotion Detection';
    } else if (stepDetails.includes('text sentiment analysis')) {
      modelName = 'testnew21/text-model';
      modelType = 'Text Sentiment Analysis';
    } else if (stepDetails.includes('audio sentiment analysis')) {
      modelName = 'testnew21/audio-model';
      modelType = 'Audio Sentiment Analysis';
    } else if (stepDetails.includes('Whisper') || stepDetails.includes('speech-to-text')) {
      modelName = 'openai/whisper-base';
      modelType = 'Speech-to-Text (Whisper)';
    }
    
    // Extract download progress from step details if available
    const progressMatch = stepDetails.match(/(\d+\.?\d*)%/);
    if (progressMatch) {
      downloadProgress = parseFloat(progressMatch[1]);
    }
    
    return {
      modelName,
      modelType,
      downloadProgress,
      isDownloading: true
    };
  };

  // Get status message based on current status
  const getStatusMessage = () => {
    if (!jobStatus) return 'Checking status...';
    
    switch (jobStatus.status) {
      case JobStatus.PENDING:
        return 'Waiting to start processing...';
      case JobStatus.PROCESSING:
        return jobStatus.current_step || 'Processing your video...';
      case JobStatus.COMPLETED:
        return 'Processing complete!';
      case JobStatus.FAILED:
        return 'Processing failed';
      default:
        return 'Checking status...';
    }
  };

  // Get progress percentage from backend or fallback
  const getProgressPercentage = () => {
    if (!jobStatus) return 0;
    
    if (jobStatus.progress_percentage !== undefined) {
      return Math.round(jobStatus.progress_percentage);
    }
    
    // Fallback to old logic if backend doesn't provide progress
    switch (jobStatus.status) {
      case JobStatus.PENDING:
        return 5;
      case JobStatus.PROCESSING:
        return Math.min(10 + (pollingCount * 5), 85);
      case JobStatus.COMPLETED:
        return 100;
      case JobStatus.FAILED:
        return 100;
      default:
        return 0;
    }
  };

  // Get step details if available
  const getStepDetails = () => {
    if (!jobStatus || !jobStatus.step_details) return null;
    return jobStatus.step_details;
  };

  // Get processing steps for visual indication
  const getProcessingSteps = () => {
    const steps = [
      'Initializing',
      'Loading Visual Model',
      'Loading Text Model',
      'Loading Audio Model',
      'Loading Whisper Model',
      'Extracting Frames',
      'Extracting Audio', 
      'Analyzing Visual Content',
      'Transcribing Audio',
      'Analyzing Audio',
      'Analyzing Text',
      'Combining Results',
      'Completed'
    ];
    
    const currentStep = jobStatus?.current_step || 'Initializing';
    const currentIndex = steps.findIndex(step => 
      currentStep.toLowerCase().includes(step.toLowerCase())
    );
    
    return steps.map((step, index) => ({
      name: step,
      completed: index <= currentIndex,
      current: index === currentIndex
    }));
  };

  const progressPercentage = getProgressPercentage();
  const stepDetails = getStepDetails();
  const processingSteps = getProcessingSteps();

  return (
    <div className="w-full">
      <div className="mb-4">
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-lg font-medium">{getStatusMessage()}</h3>
          <span className="text-sm text-gray-500">{progressPercentage}%</span>
        </div>
        
        <div className="w-full bg-gray-200 rounded-full h-3">
          <div
            className={`h-3 rounded-full transition-all duration-500 ${
              jobStatus?.status === JobStatus.FAILED ? 'bg-red-600' : 'bg-blue-600'
            }`}
            style={{ width: `${progressPercentage}%` }}
          ></div>
        </div>
        
        {stepDetails && (
          <p className="mt-2 text-sm text-gray-600 italic">{stepDetails}</p>
        )}
      </div>
      
      {/* Model Download Progress */}
      {currentModel && currentModel.isDownloading && (
        <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center mb-2">
            <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse mr-2"></div>
            <h4 className="text-sm font-medium text-blue-800">Downloading Model</h4>
          </div>
          
          <div className="mb-2">
            <div className="flex justify-between items-center text-sm">
              <span className="text-blue-700 font-medium">{currentModel.modelType}</span>
              <span className="text-blue-600">{currentModel.downloadProgress.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-blue-200 rounded-full h-2 mt-1">
              <div
                className="h-2 bg-blue-600 rounded-full transition-all duration-300"
                style={{ width: `${currentModel.downloadProgress}%` }}
              ></div>
            </div>
          </div>
          
          <p className="text-xs text-blue-600">
            Model: <code className="bg-blue-100 px-1 rounded">{currentModel.modelName}</code>
          </p>
          
          <div className="mt-2 text-xs text-blue-600">
            <p>📥 First-time downloads may take several minutes depending on your internet connection</p>
            <p>🔄 Models are cached locally for faster subsequent processing</p>
          </div>
        </div>
      )}
      
      {/* Processing Steps Visualization */}
      {jobStatus?.status === JobStatus.PROCESSING && (
        <div className="mb-4">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Processing Steps:</h4>
          <div className="space-y-1">
            {processingSteps.map((step, index) => (
              <div key={index} className="flex items-center text-sm">
                <div className={`w-3 h-3 rounded-full mr-2 ${
                  step.completed ? 'bg-green-500' : 
                  step.current ? 'bg-blue-500 animate-pulse' : 'bg-gray-300'
                }`}></div>
                <span className={`${
                  step.completed ? 'text-green-700' :
                  step.current ? 'text-blue-700 font-medium' : 'text-gray-500'
                }`}>
                  {step.name}
                  {step.current && currentModel && currentModel.isDownloading && (
                    <span className="ml-2 text-blue-600">
                      (Downloading {currentModel.modelType}...)
                    </span>
                  )}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {error && (
        <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-md">
          <p className="text-sm">{error}</p>
        </div>
      )}
      
      <div className="mt-4 text-sm text-gray-600">
        <p>This may take a few minutes depending on the video length.</p>
        <p>We are analyzing visual emotions, audio sentiment, and transcribing speech.</p>
        {jobStatus?.status === JobStatus.PROCESSING && (
          <p className="mt-1 text-blue-600">
            {stepDetails?.includes('Downloading') || stepDetails?.includes('Loading') ? 
              'First-time model downloads may take longer...' : 
              'Processing your video content...'}
          </p>
        )}
      </div>
    </div>
  );
};

export default ProcessingStatus; 