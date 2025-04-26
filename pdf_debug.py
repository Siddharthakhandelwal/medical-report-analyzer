#!/usr/bin/env python3
"""
PDF Extraction Debug Tool

This script helps diagnose issues with PDF text extraction in the Medical Report Analyzer.
It will print the full extracted text and attempt to identify age-related information.
"""

import os
import sys
import re
import PyPDF2

def extract_and_print_pdf_text(pdf_path):
    """Extract text from PDF and print it for debugging"""
    print(f"Analyzing PDF: {pdf_path}")
    print("="*80)
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return
    
    try:
        # Extract text from PDF
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            print(f"PDF has {len(reader.pages)} pages")
            
            # Extract text from each page
            for page_num in range(len(reader.pages)):
                page_text = reader.pages[page_num].extract_text()
                print(f"\n--- Page {page_num+1} Text Preview ---")
                print(page_text[:300] + "..." if len(page_text) > 300 else page_text)
                text += page_text
        
        # Try different age extraction patterns
        print("\n--- Age Detection Attempts ---")
        
        patterns = [
            r'(?:Age|DOB|Date\s+of\s+birth)[\s:]*(\d+)',  # Original pattern
            r'(?:Age|DOB|Date\s+of\s+birth)[\s:]*(\d+[\s\S]{0,5})',  # More flexible
            r'Age[\s:]*(\d+)',  # Just "Age" followed by number
            r'Age\s*:\s*(\d+)',  # Age: format
            r'Age\s*-\s*(\d+)',  # Age - format
            r'Patient\s*Age\s*:?\s*(\d+)',  # Patient Age format
            r'(?:age|AGE|Age)\s*[-:=]?\s*(\d+)',  # Case insensitive with various separators
            r'born\s+in\s+(\d{4})',  # Calculate from birth year
            r'DOB\s*[-:=]?\s*(\d+\/\d+\/\d+)',  # Date of birth pattern
            r'Date\s+of\s+Birth\s*[-:=]?\s*(\d+\/\d+\/\d+)', # Full date of birth
            r'Birth\s+Date\s*[-:=]?\s*(\d+\/\d+\/\d+)'  # Birth date alternative
        ]
        
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                print(f"Pattern {i+1}: {pattern}")
                print(f"  Matches: {matches}")
        
        # Find surrounding context for age
        print("\n--- Context Search ---")
        age_context = re.findall(r'.{0,30}(age|birth|dob|year).{0,30}', text, re.IGNORECASE)
        if age_context:
            print("Possible age-related text found:")
            for i, context in enumerate(age_context):
                print(f"  {i+1}. ...{context}...")
        
        return text
    
    except Exception as e:
        print(f"Error extracting text: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "Cheshtaa_Bhardwaj_Test.pdf"  # Default file
    
    extract_and_print_pdf_text(pdf_path)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 