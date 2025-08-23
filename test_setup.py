import os
from dotenv import load_dotenv

# Load environment variables from.env file
load_dotenv()

# Get the API key
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    print("API key loaded successfully!")
    print(f"   Your API key starts with: {api_key[:4]}...")
else:
    print("Failed to load API key. Please check your.env file.")