#!/usr/bin/env python3
"""
Simple test script to verify the upload endpoint is working.
"""

import requests
import json

def test_upload_endpoint():
    """Test the upload endpoint with a simple request."""
    url = "http://localhost:8000/api/v1/upload"
    
    # Create a simple test file (we'll just test the endpoint response)
    try:
        # Test with a simple POST request to see if the endpoint responds
        response = requests.post(url)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 422:
            print("✅ Upload endpoint is working (422 is expected without file)")
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_health_endpoint():
    """Test the health endpoint."""
    url = "http://localhost:8000/health"
    
    try:
        response = requests.get(url)
        print(f"Health Status Code: {response.status_code}")
        print(f"Health Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Health endpoint is working")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")

if __name__ == "__main__":
    print("Testing Multi-Modal Sentiment Analysis API...")
    print("=" * 50)
    
    test_health_endpoint()
    print()
    test_upload_endpoint()
    
    print("\n" + "=" * 50)
    print("Test completed!") 