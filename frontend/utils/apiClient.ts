import axios from 'axios';

// Get API base URL from environment
const API_URL = process.env.API_URL || 'http://localhost:8000';
const API_BASE_PATH = '/api/v1';

// Create axios instance
const apiClient = axios.create({
  baseURL: `${API_URL}${API_BASE_PATH}`,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API endpoints
export const endpoints = {
  upload: '/upload',
  status: (jobId: string) => `/status/${jobId}`,
  results: (jobId: string) => `/results/${jobId}`,
  cleanup: (jobId: string) => `/cleanup/${jobId}`,
};

// API client functions
export const api = {
  // Upload video file
  uploadVideo: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await apiClient.post(endpoints.upload, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  },
  
  // Get job status
  getJobStatus: async (jobId: string) => {
    const response = await apiClient.get(endpoints.status(jobId));
    return response.data;
  },
  
  // Get job results
  getJobResults: async (jobId: string) => {
    const response = await apiClient.get(endpoints.results(jobId));
    return response.data;
  },
  
  // Clean up job files
  cleanupJob: async (jobId: string) => {
    const response = await apiClient.delete(endpoints.cleanup(jobId));
    return response.data;
  },
};

export default api; 