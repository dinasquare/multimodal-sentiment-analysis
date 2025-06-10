import React, { useState } from 'react';
import Head from 'next/head';
import FileUpload from '../components/FileUpload';
import ResultsDisplay from '../components/ResultsDisplay';
import { JobResult } from '../utils/types';

enum AppStep {
  UPLOAD = 'upload',
  RESULTS = 'results',
}

export default function Home() {
  const [currentStep, setCurrentStep] = useState<AppStep>(AppStep.UPLOAD);
  const [jobResult, setJobResult] = useState<JobResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Handle successful upload and processing
  const handleUploadSuccess = (result: JobResult) => {
    setJobResult(result);
    setCurrentStep(AppStep.RESULTS);
    setError(null);
  };

  // Handle upload error
  const handleUploadError = (errorMessage: string) => {
    setError(errorMessage);
  };

  // Reset the app to the upload step
  const handleReset = () => {
    setCurrentStep(AppStep.UPLOAD);
    setJobResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>Multi-Modal Sentiment Analysis</title>
        <meta name="description" content="Analyze sentiment in videos through visual, audio, and text modalities" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <h1 className="text-3xl font-bold text-gray-900">Multi-Modal Sentiment Analysis</h1>
            {currentStep !== AppStep.UPLOAD && (
              <button
                onClick={handleReset}
                className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition"
              >
                New Analysis
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        {/* Step indicator */}
        <div className="mb-8">
          <div className="flex items-center justify-center">
            <div className={`step-item ${currentStep === AppStep.UPLOAD ? 'active' : 'completed'}`}>
              <div className="step-circle">1</div>
              <div className="step-text">Upload & Process</div>
            </div>
            <div className="step-line"></div>
            <div className={`step-item ${currentStep === AppStep.RESULTS ? 'active' : ''}`}>
              <div className="step-circle">2</div>
              <div className="step-text">Results</div>
            </div>
          </div>
        </div>

        {/* Error message */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 text-red-800 rounded-md p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Main content based on current step */}
        <div className="bg-white shadow rounded-lg p-6">
          {currentStep === AppStep.UPLOAD && (
            <div className="max-w-2xl mx-auto">
              <h2 className="text-xl font-semibold mb-4">Upload a Video for Analysis</h2>
              <p className="mb-6 text-gray-600">
                Upload a video file to analyze sentiment through visual emotions, audio tone, and speech content.
                The analysis will be completed immediately using local AI models.
              </p>
              <FileUpload
                onUploadSuccess={handleUploadSuccess}
                onUploadError={handleUploadError}
              />
            </div>
          )}

          {currentStep === AppStep.RESULTS && jobResult && (
            <ResultsDisplay jobResult={jobResult} />
          )}
        </div>
      </main>

      <footer className="bg-white border-t border-gray-200 py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <p className="text-center text-gray-500 text-sm">
            Multi-Modal Sentiment Analysis - Powered by Local AI Models
          </p>
        </div>
      </footer>

      {/* Custom CSS for step indicator */}
      <style jsx>{`
        .step-item {
          display: flex;
          flex-direction: column;
          align-items: center;
          flex: 1;
        }
        
        .step-circle {
          width: 30px;
          height: 30px;
          border-radius: 50%;
          background-color: #e5e7eb;
          color: #9ca3af;
          display: flex;
          justify-content: center;
          align-items: center;
          font-weight: bold;
        }
        
        .step-text {
          margin-top: 0.5rem;
          color: #6b7280;
          font-size: 0.875rem;
        }
        
        .step-line {
          flex: 1;
          height: 2px;
          background-color: #e5e7eb;
          margin: 0 0.5rem;
        }
        
        .step-item.active .step-circle {
          background-color: #0ea5e9;
          color: white;
        }
        
        .step-item.active .step-text {
          color: #0ea5e9;
          font-weight: 500;
        }
        
        .step-item.completed .step-circle {
          background-color: #10b981;
          color: white;
        }
      `}</style>
    </div>
  );
} 