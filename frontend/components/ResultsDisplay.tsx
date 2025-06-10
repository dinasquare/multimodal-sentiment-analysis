import React from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement } from 'chart.js';
import { Doughnut, Bar } from 'react-chartjs-2';
import { JobResult, EmotionResult } from '../utils/types';
import { jsPDF } from 'jspdf';
import 'jspdf-autotable';

// Register ChartJS components
ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement);

interface ResultsDisplayProps {
  jobResult: JobResult;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ jobResult }) => {
  // Convert sentiment score to descriptive text
  const getSentimentDescription = (score: number): string => {
    if (score >= 0.7) return 'Very Positive';
    if (score >= 0.6) return 'Positive';
    if (score >= 0.45) return 'Neutral';
    if (score >= 0.3) return 'Negative';
    return 'Very Negative';
  };

  // Get CSS class for sentiment badge
  const getSentimentClass = (score: number): string => {
    if (score >= 0.6) return 'sentiment-positive';
    if (score >= 0.4) return 'sentiment-neutral';
    return 'sentiment-negative';
  };

  // Get emotion color based on type
  const getEmotionColor = (emotion: string): string => {
    const colors: { [key: string]: string } = {
      'happy': '#22c55e',
      'joy': '#22c55e',
      'positive': '#22c55e',
      'sad': '#ef4444',
      'negative': '#ef4444',
      'angry': '#dc2626',
      'fear': '#f59e0b',
      'surprise': '#8b5cf6',
      'neutral': '#6b7280',
      'disgust': '#84cc16'
    };
    return colors[emotion.toLowerCase()] || '#6b7280';
  };

  // Download results as PDF
  const downloadPDF = () => {
    if (!jobResult || !jobResult.results) return;

    const doc = new jsPDF();
    const r = jobResult.results;

    // Add title
    doc.setFontSize(20);
    doc.text('Sentiment Analysis Results', 105, 15, { align: 'center' });
    
    // Add file info
    doc.setFontSize(12);
    doc.text(`File: ${jobResult.original_filename}`, 20, 30);
    doc.text(`Analysis Date: ${new Date(jobResult.updated_at).toLocaleString()}`, 20, 38);
    
    // Add combined score
    doc.setFontSize(16);
    doc.text('Combined Sentiment Score', 105, 50, { align: 'center' });
    doc.setFontSize(14);
    doc.text(
      `${(r.combined_sentiment * 100).toFixed(1)}% (${getSentimentDescription(r.combined_sentiment)})`,
      105, 58, { align: 'center' }
    );
    doc.text(`Confidence: ${(r.confidence * 100).toFixed(1)}%`, 105, 66, { align: 'center' });
    
    // Add individual scores
    doc.setFontSize(16);
    doc.text('Individual Modality Scores', 105, 80, { align: 'center' });
    
    const tableData = [
      ['Modality', 'Score', 'Sentiment'],
      ['Visual', `${(r.visual_sentiment || 0 * 100).toFixed(1)}%`, getSentimentDescription(r.visual_sentiment || 0)],
      ['Audio', `${(r.audio_sentiment || 0 * 100).toFixed(1)}%`, getSentimentDescription(r.audio_sentiment || 0)],
      ['Text', `${(r.text_sentiment || 0 * 100).toFixed(1)}%`, getSentimentDescription(r.text_sentiment || 0)],
    ];
    
    // @ts-ignore - jspdf-autotable types are not available
    doc.autoTable({
      startY: 85,
      head: [tableData[0]],
      body: tableData.slice(1),
      theme: 'striped',
      headStyles: { fillColor: [14, 165, 233] },
    });
    
    // Save PDF
    doc.save(`sentiment-analysis-${jobResult.id}.pdf`);
  };

  // Prepare chart data for combined sentiment
  const prepareCombinedChartData = () => {
    if (!jobResult || !jobResult.results) return null;
    
    const r = jobResult.results;
    return {
      labels: ['Positive', 'Negative'],
      datasets: [
        {
          data: [r.combined_sentiment, 1 - r.combined_sentiment],
          backgroundColor: ['#22c55e', '#ef4444'],
          borderColor: ['#16a34a', '#dc2626'],
          borderWidth: 1,
        },
      ],
    };
  };

  // Prepare chart data for modality comparison with better labels
  const prepareModalityChartData = () => {
    if (!jobResult || !jobResult.results) return null;
    
    const r = jobResult.results;
    return {
      labels: [
        'Visual Emotions\n(Facial Analysis)', 
        'Audio Sentiment\n(Voice Tone)', 
        'Text Sentiment\n(Speech Content)'
      ],
      datasets: [
        {
          label: 'Sentiment Score',
          data: [
            r.visual_sentiment || 0,
            r.audio_sentiment || 0,
            r.text_sentiment || 0,
          ],
          backgroundColor: [
            'rgba(53, 162, 235, 0.8)',
            'rgba(75, 192, 192, 0.8)',
            'rgba(255, 206, 86, 0.8)',
          ],
          borderColor: [
            'rgba(53, 162, 235, 1)',
            'rgba(75, 192, 192, 1)',
            'rgba(255, 206, 86, 1)',
          ],
          borderWidth: 2,
        },
      ],
    };
  };

  // Get transcribed text from jobResult
  const getTranscribedText = (): string => {
    // Check if we have transcription in the results
    if (jobResult?.results && 'transcription' in jobResult.results) {
      return (jobResult.results as any).transcription || '';
    }
    return '';
  };

  // Get visual emotions data
  const getVisualEmotionsData = () => {
    if (jobResult?.results && 'visual_emotions' in jobResult.results) {
      return (jobResult.results as any).visual_emotions || [];
    }
    return [];
  };

  // Get dominant visual emotion
  const getDominantVisualEmotion = (): string => {
    if (jobResult?.results && 'dominant_visual_emotion' in jobResult.results) {
      return (jobResult.results as any).dominant_visual_emotion || 'neutral';
    }
    return 'neutral';
  };

  if (!jobResult || !jobResult.results) {
    return (
      <div className="bg-yellow-50 p-4 rounded-md">
        <h3 className="text-lg font-medium text-yellow-800">No results available</h3>
        <p className="mt-1 text-sm text-yellow-700">
          The analysis results are not available. This could be due to an error during processing.
        </p>
      </div>
    );
  }

  const r = jobResult.results;
  const combinedChartData = prepareCombinedChartData();
  const modalityChartData = prepareModalityChartData();
  const transcribedText = getTranscribedText();
  const visualEmotions = getVisualEmotionsData();
  const dominantEmotion = getDominantVisualEmotion();

  return (
    <div className="space-y-8">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Analysis Results</h2>
        <button
          onClick={downloadPDF}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition"
        >
          Download PDF
        </button>
      </div>

      {/* Transcribed Text Section */}
      {transcribedText && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold mb-4 flex items-center">
            <svg className="w-5 h-5 mr-2 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4V2a1 1 0 011-1h8a1 1 0 011 1v2m-9 0h10m-10 0a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V6a2 2 0 00-2-2" />
            </svg>
            Transcribed Speech
          </h3>
          <div className="bg-gray-50 rounded-lg p-4 border-l-4 border-blue-500">
            <p className="text-gray-800 italic">"{transcribedText}"</p>
          </div>
        </div>
      )}

      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-semibold mb-4">Combined Sentiment</h3>
        <div className="flex flex-col md:flex-row items-center justify-around">
          <div className="w-48 h-48">
            {combinedChartData && <Doughnut data={combinedChartData} />}
          </div>
          <div className="mt-4 md:mt-0 text-center">
            <div className="text-4xl font-bold">{(r.combined_sentiment * 100).toFixed(1)}%</div>
            <div className={`mt-2 sentiment-badge ${getSentimentClass(r.combined_sentiment)}`}>
              {getSentimentDescription(r.combined_sentiment)}
            </div>
            <div className="mt-2 text-sm text-gray-500">
              Confidence: {(r.confidence * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-semibold mb-4">Modality Comparison</h3>
        <div className="h-64">
          {modalityChartData && <Bar 
            data={modalityChartData}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: {
                  display: true,
                  position: 'top' as const,
                },
                tooltip: {
                  callbacks: {
                    label: function(context) {
                      const value = (Number(context.parsed.y) * 100).toFixed(1);
                      return `${context.dataset.label}: ${value}%`;
                    }
                  }
                }
              },
              scales: {
                x: {
                  ticks: {
                    maxRotation: 0,
                    font: {
                      size: 11
                    }
                  }
                },
                y: {
                  beginAtZero: true,
                  max: 1,
                  ticks: {
                    callback: function(value) {
                      return (Number(value) * 100) + '%';
                    }
                  },
                  title: {
                    display: true,
                    text: 'Sentiment Score (%)'
                  }
                }
              }
            }}
          />}
        </div>
        <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-600">
          <div className="text-center">
            <div className="font-medium text-blue-600">Visual Emotions</div>
            <div>Analyzes facial expressions and emotions from video frames</div>
          </div>
          <div className="text-center">
            <div className="font-medium text-teal-600">Audio Sentiment</div>
            <div>Analyzes voice tone, pitch, and audio characteristics</div>
          </div>
          <div className="text-center">
            <div className="font-medium text-yellow-600">Text Sentiment</div>
            <div>Analyzes sentiment from transcribed speech content</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Visual Emotions */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <svg className="w-5 h-5 mr-2 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
            Visual Emotions
          </h3>
          <div className="space-y-3">
            <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
              <div className="text-center">
                <div className="text-2xl mb-2">😊</div>
                <div className="font-medium text-blue-800">Dominant Emotion</div>
                <div className="text-lg font-bold capitalize" style={{color: getEmotionColor(dominantEmotion)}}>
                  {dominantEmotion}
                </div>
                <div className="text-sm text-gray-600 mt-2">
                  Detected from facial expressions
                </div>
              </div>
            </div>
            {visualEmotions.length > 0 && (
              <div className="space-y-2">
                <div className="text-sm font-medium text-gray-700">Frame Analysis:</div>
                {visualEmotions.slice(0, 3).map((emotion: any, index: number) => (
                  <div key={index} className="flex justify-between items-center text-sm">
                    <span className="capitalize">{emotion.emotion}</span>
                    <span className="font-medium">{(emotion.confidence * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Audio Sentiment */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <svg className="w-5 h-5 mr-2 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
            </svg>
            Audio Sentiment
          </h3>
          <div className="space-y-3">
            <div className="bg-teal-50 rounded-lg p-4 border border-teal-200">
              <div className="text-center">
                <div className="text-2xl mb-2">🎵</div>
                <div className="font-medium text-teal-800">Voice Tone Analysis</div>
                <div className="text-lg font-bold" style={{color: getEmotionColor(getSentimentDescription(r.audio_sentiment || 0))}}>
                  {getSentimentDescription(r.audio_sentiment || 0)}
                </div>
                <div className="text-sm text-gray-600 mt-2">
                  Score: {((r.audio_sentiment || 0) * 100).toFixed(1)}%
                </div>
              </div>
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Energy Level:</span>
                <span className="font-medium">
                  {(r.audio_sentiment || 0) > 0.6 ? 'High' : (r.audio_sentiment || 0) > 0.4 ? 'Medium' : 'Low'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Tone Quality:</span>
                <span className="font-medium">
                  {(r.audio_sentiment || 0) > 0.6 ? 'Positive' : (r.audio_sentiment || 0) > 0.4 ? 'Neutral' : 'Negative'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Text Sentiment */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <svg className="w-5 h-5 mr-2 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
            </svg>
            Text Sentiment
          </h3>
          <div className="space-y-3">
            <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
              <div className="text-center">
                <div className="text-2xl mb-2">💬</div>
                <div className="font-medium text-yellow-800">Speech Content</div>
                <div className="text-lg font-bold" style={{color: getEmotionColor(getSentimentDescription(r.text_sentiment || 0))}}>
                  {getSentimentDescription(r.text_sentiment || 0)}
                </div>
                <div className="text-sm text-gray-600 mt-2">
                  Score: {((r.text_sentiment || 0) * 100).toFixed(1)}%
                </div>
              </div>
            </div>
            {transcribedText && (
              <div className="space-y-2 text-sm">
                <div className="text-gray-600">Word Analysis:</div>
                <div className="bg-gray-50 rounded p-2">
                  <div className="flex justify-between">
                    <span>Positive indicators:</span>
                    <span className="text-green-600 font-medium">
                      {transcribedText.toLowerCase().split(' ').filter(word => 
                        ['good', 'great', 'happy', 'positive', 'excellent', 'wonderful', 'love', 'joy', 'amazing', 'fantastic'].includes(word)
                      ).length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Negative indicators:</span>
                    <span className="text-red-600 font-medium">
                      {transcribedText.toLowerCase().split(' ').filter(word => 
                        ['bad', 'terrible', 'sad', 'negative', 'awful', 'hate', 'anger', 'fear', 'horrible', 'disgusting'].includes(word)
                      ).length}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Custom CSS for sentiment badges */}
      <style jsx>{`
        .sentiment-badge {
          display: inline-block;
          padding: 0.5rem 1rem;
          border-radius: 0.5rem;
          font-weight: 600;
          font-size: 0.875rem;
        }
        
        .sentiment-positive {
          background-color: #dcfce7;
          color: #166534;
        }
        
        .sentiment-neutral {
          background-color: #f3f4f6;
          color: #374151;
        }
        
        .sentiment-negative {
          background-color: #fecaca;
          color: #991b1b;
        }
      `}</style>
    </div>
  );
};

export default ResultsDisplay; 