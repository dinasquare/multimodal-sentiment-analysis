import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { JobResult } from '../utils/types';

interface FileUploadProps {
  onUploadSuccess: (result: JobResult) => void;
  onUploadError: (error: string) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onUploadSuccess, onUploadError }) => {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingStatus, setProcessingStatus] = useState('');

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    
    // Validate file size (max 100MB)
    const maxSize = 100 * 1024 * 1024; // 100MB
    if (file.size > maxSize) {
      onUploadError('File size must be less than 100MB');
      return;
    }

    // Validate file type
    const allowedTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/quicktime'];
    if (!allowedTypes.includes(file.type)) {
      onUploadError('Please upload a valid video file (MP4, AVI, MOV)');
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);
    setProcessingStatus('Preparing upload...');

    try {
      const formData = new FormData();
      formData.append('file', file);

      // Create XMLHttpRequest for progress tracking
      const xhr = new XMLHttpRequest();
      
      // Set timeout to 10 minutes (video processing can take time)
      xhr.timeout = 10 * 60 * 1000; // 10 minutes
      
      // Track upload progress
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const progress = Math.round((event.loaded / event.total) * 100);
          setUploadProgress(progress);
          if (progress < 100) {
            setProcessingStatus('Uploading video...');
          } else {
            setProcessingStatus('Processing video with AI models...');
          }
        }
      });

      // Handle response
      xhr.addEventListener('load', () => {
        if (xhr.status === 200) {
          try {
            const result: JobResult = JSON.parse(xhr.responseText);
            setProcessingStatus('Analysis complete!');
            onUploadSuccess(result);
          } catch (error) {
            onUploadError('Invalid response from server');
          }
        } else {
          try {
            const errorData = JSON.parse(xhr.responseText);
            onUploadError(errorData.detail || 'Upload failed');
          } catch {
            onUploadError(`Upload failed with status ${xhr.status}`);
          }
        }
        setIsUploading(false);
        setUploadProgress(0);
        setProcessingStatus('');
      });

      // Handle errors
      xhr.addEventListener('error', () => {
        onUploadError('Network error occurred during upload');
        setIsUploading(false);
        setUploadProgress(0);
        setProcessingStatus('');
      });

      // Handle timeout
      xhr.addEventListener('timeout', () => {
        onUploadError('Request timed out. Video processing is taking longer than expected. Please try with a shorter video.');
        setIsUploading(false);
        setUploadProgress(0);
        setProcessingStatus('');
      });

      // Handle abort
      xhr.addEventListener('abort', () => {
        onUploadError('Upload was cancelled');
        setIsUploading(false);
        setUploadProgress(0);
        setProcessingStatus('');
      });

      // Send request
      xhr.open('POST', '/api/v1/upload');
      xhr.send(formData);

      // Update status after upload completes
      xhr.upload.addEventListener('loadend', () => {
        if (uploadProgress >= 100) {
          setProcessingStatus('Analyzing video with AI models... This may take 1-3 minutes.');
        }
      });

    } catch (error) {
      console.error('Upload error:', error);
      onUploadError('An unexpected error occurred');
      setIsUploading(false);
      setUploadProgress(0);
      setProcessingStatus('');
    }
  }, [onUploadSuccess, onUploadError]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov']
    },
    multiple: false,
    disabled: isUploading
  });

  return (
    <div className="w-full">
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${isDragActive ? 'border-blue-400 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}
          ${isUploading ? 'cursor-not-allowed opacity-50' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        {!isUploading ? (
          <>
            <svg
              className="mx-auto h-12 w-12 text-gray-400 mb-4"
              stroke="currentColor"
              fill="none"
              viewBox="0 0 48 48"
            >
              <path
                d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                strokeWidth={2}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            
            {isDragActive ? (
              <p className="text-blue-600 font-medium">Drop the video file here</p>
            ) : (
              <>
                <p className="text-gray-600 font-medium mb-2">
                  Drag and drop a video file here, or click to select
                </p>
                <p className="text-sm text-gray-500">
                  Supported formats: MP4, AVI, MOV (max 100MB)
                </p>
                <p className="text-xs text-gray-400 mt-2">
                  First analysis may take 1-3 minutes while AI models load
                </p>
              </>
            )}
          </>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            </div>
            
            <div>
              <p className="text-blue-600 font-medium mb-2">
                {processingStatus || 'Processing...'}
              </p>
              
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
              
              <p className="text-sm text-gray-500 mt-2">
                {uploadProgress < 100 ? `${uploadProgress}% uploaded` : 'Processing with AI models...'}
              </p>
            </div>
            
            <div className="text-sm text-gray-500 space-y-1">
              <p>🔄 Loading AI models (Visual, Audio, Text, Speech-to-Text)</p>
              <p>🎬 Extracting frames and audio from video</p>
              <p>🤖 Analyzing sentiment across all modalities</p>
              <p className="text-xs text-gray-400 mt-2">
                Please be patient - this process ensures accurate results!
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUpload; 