#!/usr/bin/env python3
"""
PDF Information Extractor

This utility helps extract and visualize key medical information from PDF files.
It can be used to debug extraction issues and see what values are being found.
"""

import os
import sys
import PyPDF2
import re
from datetime import datetime
from medical_report_analyzer import MedicalReportAnalyzer

def highlight_matches(text, pattern, case_insensitive=True):
    """Highlight matches of a pattern in the text"""
    flags = re.IGNORECASE if case_insensitive else 0
    matches = list(re.finditer(pattern, text, flags))
    
    if not matches:
        return []
    
    results = []
    for match in matches:
        start, end = match.span()
        context_start = max(0, start - 30)
        context_end = min(len(text), end + 30)
        
        prefix = "..." if context_start > 0 else ""
        suffix = "..." if context_end < len(text) else ""
        
        # Extract the matched value and surrounding context
        context = text[context_start:start]
        matched = text[start:end]
        post_context = text[end:context_end]
        
        results.append({
            "context_before": prefix + context,
            "match": matched,
            "context_after": post_context + suffix,
            "value": match.group(1) if match.groups() else match.group(0)
        })
    
    return results

def extract_info_from_pdf(pdf_path):
    """Extract all relevant information from a PDF"""
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return None
    
    try:
        # Use the analyzer to extract text
        analyzer = MedicalReportAnalyzer()
        text = analyzer.extract_text_from_pdf(pdf_path)
        
        if not text:
            print("Error: Failed to extract text from PDF")
            return None
        
        # Extract medical values using regex alone
        print("\n" + "="*50)
        print("REGEX-ONLY EXTRACTION")
        print("="*50)
        regex_data = analyzer._extract_with_regex(text)
        for key, value in regex_data.items():
            print(f"{key}: {value}")
        
        # Extract using Gemini if available
        if hasattr(analyzer, '_extract_with_gemini'):
            print("\n" + "="*50)
            print("GEMINI AI EXTRACTION")
            print("="*50)
            try:
                ai_data = analyzer._extract_with_gemini(text)
                for key, value in ai_data.items():
                    print(f"{key}: {value}")
            except Exception as e:
                print(f"AI extraction error: {e}")
        
        # Extract with the full pipeline (regex + AI)
        print("\n" + "="*50)
        print(f"COMBINED EXTRACTION FROM: {pdf_path}")
        print("="*50)
        medical_data = analyzer.extract_medical_values(text)
        for key, value in medical_data.items():
            print(f"{key}: {value}")
        
        # Highlight key patterns in the text
        key_patterns = {
            "Age": r'(?:Patient\s+age|Patient\'s\s+age|Age\s+of\s+patient|Years\s+Old|Patient\s+is\s+(\d+)\s+years|(\d+)[-\s]year[-\s]old|Age[:]\s*(\d+))',
            "Glucose": r'(?:Glucose|Blood\s+glucose|Sugar|Blood\s+Sugar)[\s:]*(\d+\.?\d*)',
            "Blood Pressure": r'(?:Blood\s+pressure|BP)[\s:]*(\d+/\d+)',
            "Cholesterol": r'(?:Cholesterol|Total\s+cholesterol)[\s:]*(\d+\.?\d*)',
            "BMI": r'(?:BMI|Body\s+Mass\s+Index)[\s:]*(\d+\.?\d*)'
        }
        
        print("\n" + "="*50)
        print("PATTERN MATCHES WITH CONTEXT")
        print("="*50)
        
        for key, pattern in key_patterns.items():
            matches = highlight_matches(text, pattern)
            if matches:
                print(f"\n{key} - Pattern: {pattern}")
                for i, match in enumerate(matches):
                    print(f"  Match {i+1}: {match['context_before']}[{match['match']}]{match['context_after']}")
                    print(f"    Extracted value: {match['value']}")
            else:
                print(f"\n{key} - No matches found with pattern: {pattern}")
        
        # Analyze the report
        results = analyzer.analyze_report(pdf_path)
        
        print("\n" + "="*50)
        print("ANALYSIS RESULTS")
        print("="*50)
        print(f"Diabetes Risk: {'High' if results['diabetes_analysis'].get('has_diabetes', False) else 'Low'}")
        print(f"Heart Disease Risk: {'High' if results['heart_disease_analysis'].get('has_heart_disease', False) else 'Low'}")
        
        return {
            'text': text, 
            'regex_data': regex_data,
            'ai_data': ai_data if 'ai_data' in locals() else None,
            'combined_data': medical_data,
            'results': results
        }
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("Enter the path to the PDF file: ")
    
    extract_info_from_pdf(pdf_path)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 