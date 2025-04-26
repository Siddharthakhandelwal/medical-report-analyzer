#!/usr/bin/env python3
"""
Gemini API Diagnostic Tool

This script helps diagnose issues with the Google Gemini AI API.
It lists available models and checks if they can be used for generating content.

Usage:
    python gemini_diagnostic.py
"""

import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai

def main():
    """Main function to diagnose Gemini API issues"""
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please create a .env file with your API key or set it as an environment variable.")
        return 1
    
    # Configure Gemini API
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        return 1
    
    print("=== Gemini API Diagnostic Tool ===\n")
    
    # List available models
    try:
        print("Fetching available models...")
        models = genai.list_models()
        print("\nAvailable models:")
        
        generation_models = []
        
        for model in models:
            model_info = f"- {model.name}"
            
            # Check if the model supports content generation
            if "generateContent" in model.supported_generation_methods:
                model_info += " (Supports generateContent)"
                generation_models.append(model.name)
            
            print(model_info)
        
        if generation_models:
            print("\nModels that support content generation:")
            for model in generation_models:
                print(f"- {model}")
        else:
            print("\nNo models found that support content generation.")
        
        # Test a sample generation
        if generation_models:
            print("\nTesting content generation with first available model...")
            model_name = generation_models[0]
            print(f"Using model: {model_name}")
            
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("Write a one-sentence test response.")
                print(f"Test response: {response.text}")
                print("\nContent generation working correctly!")
            except Exception as e:
                print(f"Error generating content: {e}")
                print("\nTroubleshooting suggestions:")
                print("1. Check your API key is correct and has proper permissions")
                print("2. Make sure your account has access to the Gemini API")
                print("3. Check for any regional restrictions")
        
        print("\nTo update the model in medical_report_analyzer.py:")
        print("1. Open the file and find the generate_ai_report method")
        print("2. Update the model name to one of the supported models above")
        print("3. Save the file and try again")
        
        return 0
    
    except Exception as e:
        print(f"Error accessing Gemini API: {e}")
        print("\nTroubleshooting suggestions:")
        print("1. Check your internet connection")
        print("2. Verify that your API key is correct")
        print("3. Check if the Gemini API is available in your region")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 