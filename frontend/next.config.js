/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  
  // API configuration for long-running requests
  async rewrites() {
    return [
      {
        source: '/api/v1/:path*',
        destination: 'http://localhost:8000/api/v1/:path*',
      },
    ];
  },
  
  // Increase timeout for API routes
  experimental: {
    proxyTimeout: 10 * 60 * 1000, // 10 minutes
  },
  
  // Server configuration
  serverRuntimeConfig: {
    // Increase body size limit for video uploads
    maxFileSize: 100 * 1024 * 1024, // 100MB
  },
  
  // Public runtime config
  publicRuntimeConfig: {
    apiUrl: process.env.API_URL || 'http://localhost:8000',
  },
};

module.exports = nextConfig; 