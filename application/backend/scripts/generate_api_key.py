#!/usr/bin/env python3
"""
API Key Generator
Generate secure API keys for authentication
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.security import generate_api_key, hash_api_key


def main():
    """Generate and display new API key"""
    print("=" * 60)
    print("API Key Generator")
    print("=" * 60)
    
    # Generate key
    api_key = generate_api_key()
    hashed = hash_api_key(api_key)
    
    print(f"\nGenerated API Key:")
    print(f"  Plain:  {api_key}")
    print(f"  Hash:   {hashed}")
    
    print(f"\n\nAdd to .env file:")
    print(f"  API_KEYS='{api_key}'")
    
    print(f"\n\nUse in requests:")
    print(f"  curl -H 'X-API-Key: {api_key}' http://localhost:8000/api/v1/health")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
