# Medical Report Analyzer - Handover Document

## Overview

This document provides essential information for testing the Medical Report Analyzer application. The application analyzes medical reports (in PDF format) to predict the risk of diabetes and heart disease, and provides personalized health recommendations using artificial intelligence.

## Setup Instructions

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up API Key**:

   - Get a Google Gemini API key from https://ai.google.dev/
   - Rename `env.template` to `.env`
   - Add your API key to the file: `GOOGLE_API_KEY=your_api_key_here`

3. **Launch the Application**:
   ```bash
   python run.py
   ```

## Testing Scenarios

### Scenario 1: Basic Analysis

1. Launch the application with `python run.py`
2. Click the "Browse" button
3. Select one of the sample PDF files (e.g., `Cheshtaa_Bhardwaj_Test.pdf` or `sterling-accuris-pathology-sample-report-unlocked.pdf`)
4. Click "Analyze Report"
5. Verify:
   - The Summary tab shows risk assessments
   - The AI Recommendations tab contains personalized advice
   - The Extracted Data tab displays medical values from the report

### Scenario 2: Sample Report Generation

1. Generate a sample medical report:
   ```bash
   python generate_sample_report.py --output test_report.pdf
   ```
2. Analyze this report to see how the system works with known values
3. Try different values to test edge cases:

   ```bash
   # High risk report
   python generate_sample_report.py --glucose 180 --cholesterol 240 --output high_risk.pdf

   # Low risk report
   python generate_sample_report.py --glucose 90 --cholesterol 180 --output low_risk.pdf
   ```

### Scenario 3: Extraction Debugging

If you encounter issues with the extraction:

1. Run the PDF info extractor on your file:
   ```bash
   python pdf_info_extractor.py path/to/report.pdf
   ```
2. Examine:
   - Which values are extracted by regex patterns
   - Which values are extracted by the AI
   - Which patterns match and in what context
   - The final combined results

### Scenario 4: API Diagnostic

Test the Gemini API functionality:

```bash
python gemini_diagnostic.py
```

This will show available models and verify if content generation is working correctly.

## Common Issues & Solutions

### PDF Extraction Problems

- **Text not extracted correctly**: Some PDFs may have security settings or use image-based content that can't be extracted. Try using the pdf_info_extractor.py to see what text is being extracted.

### Missing Medical Values

- **Values not detected**: If specific medical values aren't being detected, the pdf_info_extractor.py tool will show which extraction method is failing and why.

### AI Integration Issues

- **API Key errors**: Ensure your API key is correctly set in the .env file
- **Model availability**: Some Gemini models may not be available in all regions. Use gemini_diagnostic.py to check which models are available to you.

## Files and Their Purpose

- **run.py**: The main application entry point
- **medical_report_analyzer.py**: Core functionality for extraction and analysis
- **pdf_info_extractor.py**: Debugging tool for PDF extraction
- **generate_sample_report.py**: Tool to create sample reports with known values
- **gemini_diagnostic.py**: Tool to check Gemini AI functionality
- **test_pdf_extraction.py**: Test script for PDF extraction

## Notes for Testing

1. The hybrid extraction approach uses both pattern matching and AI to improve accuracy
2. Extraction is the most challenging part of the system - focus testing on different PDF formats
3. The AI recommendations may vary slightly each time due to the nature of generative AI
4. Pre-trained models are included, so no model training is needed during testing

Please report any issues or feedback including:

- PDF types that don't extract correctly
- Medical values that aren't being detected
- User interface improvements
- Any crashes or unexpected behavior
