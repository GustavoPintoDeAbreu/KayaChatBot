"""
Minimal Azure OpenAI test to verify credentials and quota
"""

import json
import os
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv

def load_credentials():
    """Load Azure credentials from environment variables"""
    # Load from .env file
    load_dotenv()
    
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', 'https://kaya-openai.openai.azure.com/')
    
    return api_key, endpoint

def main():
    print("Testing Azure OpenAI connection...")
    
    api_key, endpoint = load_credentials()
    
    if not api_key:
        print("❌ API key not found in .env file")
        print("   Create .env from .env.template and add your credentials")
        return
    
    print(f"✅ Found API key")
    print(f"✅ Endpoint: {endpoint}")
    
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-12-01-preview",
        azure_endpoint=endpoint
    )
    
    print("\nSending test request...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": "Say 'Hello from Azure' in Portuguese"}
            ],
            max_tokens=50
        )
        
        result = response.choices[0].message.content
        print(f"✅ Response: {result}")
        print("\n🎉 Azure OpenAI is working!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if '429' in str(e):
            print("\n⚠️  Rate limit hit - your Azure quota may be exhausted")
            print("   Wait a few minutes or check Azure portal for quota")

if __name__ == "__main__":
    main()
