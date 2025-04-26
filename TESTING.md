# Medical Report Analyzer - Testing Guide

This guide provides simple steps to test the Medical Report Analyzer application after simplification.

## Quick Start

1. Make sure Python and required dependencies are installed:

   ```
   pip install -r requirements.txt
   ```

2. Set up your Google API key in a `.env` file:

   ```
   GOOGLE_API_KEY=your_key_here
   ```

3. Launch the application:
   ```
   python run.py
   ```

## End-to-End Testing

### Option 1: Using Sample PDF Reports

The project includes two sample medical report PDFs:

- `Cheshtaa_Bhardwaj_Test.pdf`
- `sterling-accuris-pathology-sample-report-unlocked.pdf`

1. Launch the application:

   ```
   python run.py
   ```

2. Click the "Browse" button and select one of the sample PDF files.

3. Click "Analyze Report" to process the report.

4. Check the three tabs to review results:
   - **Summary**: Overall risk assessment
   - **AI Recommendations**: Detailed health recommendations
   - **Extracted Data**: Raw medical data extracted from the report

### Option 2: Generate a Test Report

1. Generate a sample medical report with test data:

   ```
   python generate_sample_report.py --output test_report.pdf
   ```

   You can customize values:

   ```
   python generate_sample_report.py --glucose 160 --cholesterol 240 --output high_risk_report.pdf
   ```

2. Launch the application and analyze this report:

   ```
   python run.py
   ```

3. Browse to select the generated report and analyze it.

### Option 3: Testing PDF Extraction

To test only the PDF extraction component:

```
python test_pdf_extraction.py
```

This will create a sample PDF report, extract the values, and verify the extraction accuracy.

## Troubleshooting Gemini AI

If you encounter issues with AI recommendations, run:

```
python gemini_diagnostic.py
```

This will check your API key and list available Gemini models that can be used.
