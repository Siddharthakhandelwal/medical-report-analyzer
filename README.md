# Medical Report Analyzer

![Medical Report Analyzer Banner](https://raw.githubusercontent.com/username/medical-report-analyzer/main/docs/images/banner.png)

This application analyzes medical reports (in PDF format) to predict the risk of diabetes and heart disease, and provides personalized health recommendations using artificial intelligence.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Local Setup](#local-setup)
  - [Setting Up Your API Key](#setting-up-your-api-key)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [Analyzing a Medical Report](#analyzing-a-medical-report)
  - [Generating Sample Reports](#generating-sample-reports)
- [Testing](#testing)
- [How It Works](#how-it-works)
  - [Extraction Process](#extraction-process)
  - [Risk Assessment](#risk-assessment)
  - [AI Recommendations](#ai-recommendations)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- **PDF Text Extraction**: Extract text data from medical report PDFs
- **Medical Value Detection**: Automatically identify key health indicators like glucose levels, blood pressure, etc.
- **Disease Risk Assessment**: Predict the risk of diabetes and heart disease
- **AI-Powered Recommendations**: Generate personalized diet, exercise, and lifestyle recommendations
- **Hybrid Extraction**: Uses both regex patterns and AI to extract medical values
- **Pre-trained Models**: Includes machine learning models for faster analysis

## Project Structure

```
medical-report-analyzer/
├── run.py                      # Main entry point for the application
├── medical_report_analyzer.py  # Core functionality with GUI implementation
├── pdf_info_extractor.py       # Tool for debugging PDF extraction issues
├── pdf_debug.py                # Debug utility for PDF text extraction
├── gemini_diagnostic.py        # Tool to diagnose Gemini AI functionality
├── generate_sample_report.py   # Utility for creating sample medical reports
├── test_pdf_extraction.py      # Test script for PDF extraction capabilities
├── freeze_models.py            # Script for training and saving ML models
├── requirements.txt            # Required Python packages
├── TESTING.md                  # Testing guide
├── HANDOVER.md                 # Handover document for testers
├── env.template                # Template for .env file with API key
├── .gitignore                  # Git ignore file
├── models/                     # Pre-trained machine learning models
│   ├── diabetes_model.pkl      # SVM model for diabetes prediction
│   ├── diabetes_scaler.pkl     # StandardScaler for diabetes data
│   ├── heart_disease_model.pkl # Logistic Regression model for heart disease
│   └── heart_scaler.pkl        # StandardScaler for heart disease data
├── data/                       # Training data
│   ├── diabetes.csv            # Dataset for diabetes model
│   └── heart_disease_data.csv  # Dataset for heart disease model
└── sample_reports/             # Sample medical reports for testing
    ├── Cheshtaa_Bhardwaj_Test.pdf
    └── sterling-accuris-pathology-sample-report-unlocked.pdf
```

### Key Files and Their Functions

| File                         | Description                                              |
| ---------------------------- | -------------------------------------------------------- |
| `run.py`                     | Main entry point that launches the GUI application       |
| `medical_report_analyzer.py` | Contains the core analyzer class and GUI implementation  |
| `pdf_info_extractor.py`      | Detailed extraction tool for debugging extraction issues |
| `gemini_diagnostic.py`       | Utility to check and diagnose Gemini AI functionality    |
| `generate_sample_report.py`  | Tool to create custom medical reports for testing        |
| `test_pdf_extraction.py`     | Tests PDF extraction functionality                       |
| `freeze_models.py`           | Script for training and saving the ML models             |

## Installation

### Prerequisites

- Python 3.7 or higher
- Pip package manager
- Google Gemini API key (for AI recommendations)

### Local Setup

1. **Clone the repository**

```bash
git clone https://github.com/siddharthakhandelwal/medical-report-analyzer.git
cd medical-report-analyzer
```

2. **Install required dependencies**

```bash
pip install -r requirements.txt
```

> **Note for Windows users**: If you encounter issues with tkinter, it comes pre-installed with Python on Windows. If it's not working, try reinstalling Python and check the option to install tcl/tk and IDLE.

<details>
<summary>Installation Screenshots (Click to expand)</summary>

![Clone Repository](https://raw.githubusercontent.com/username/medical-report-analyzer/main/docs/images/clone.png)

![Install Requirements](https://raw.githubusercontent.com/username/medical-report-analyzer/main/docs/images/install-requirements.png)

</details>

### Setting Up Your API Key

1. Get a Google API key from [Google AI Studio](https://ai.google.dev/)
2. Rename `env.template` to `.env`
3. Add your API key to the file:

```
GOOGLE_API_KEY=your_api_key_here
```

<details>
<summary>API Key Setup Screenshots (Click to expand)</summary>

![Get API Key](https://raw.githubusercontent.com/username/medical-report-analyzer/main/docs/images/api-key.png)

![Set Environment](https://raw.githubusercontent.com/username/medical-report-analyzer/main/docs/images/env-setup.png)

</details>

## Usage

### Running the Application

To start the application, run:

```bash
python run.py
```

This will launch the graphical user interface:

![Application Screenshot](https://raw.githubusercontent.com/username/medical-report-analyzer/main/docs/images/app-screenshot.png)

### Analyzing a Medical Report

1. Click the "Browse" button to select a medical report PDF file
2. Click "Analyze Report" to process the report
3. View the results in the three tabs:
   - **Summary**: Overall risk assessment for diabetes and heart disease
   - **AI Recommendations**: Detailed health recommendations
   - **Extracted Data**: Raw medical data extracted from the report

<details>
<summary>Analysis Screenshots (Click to expand)</summary>

![Select Report](https://raw.githubusercontent.com/username/medical-report-analyzer/main/docs/images/select-report.png)

![Analysis Results](https://raw.githubusercontent.com/username/medical-report-analyzer/main/docs/images/analysis-results.png)

</details>

### Generating Sample Reports

You can generate sample medical reports with custom values for testing:

```bash
# Create a basic sample report
python generate_sample_report.py --output test_report.pdf

# Create a high-risk sample
python generate_sample_report.py --glucose 180 --cholesterol 240 --output high_risk.pdf

# Create a low-risk sample
python generate_sample_report.py --glucose 90 --cholesterol 180 --output low_risk.pdf
```

Available parameters:

- `--name`: Patient name
- `--age`: Patient age
- `--glucose`: Blood glucose level
- `--bp`: Blood pressure (systolic/diastolic)
- `--insulin`: Insulin level
- `--bmi`: Body Mass Index
- `--cholesterol`: Total cholesterol
- `--output`: Output file path

## Testing

See `TESTING.md` for detailed testing instructions, including:

- Using sample PDF reports
- Generating custom test reports
- Testing with your own medical reports

For detailed debugging of PDF extraction issues:

```bash
python pdf_info_extractor.py path/to/your/medical_report.pdf
```

This will show:

- Extracted text from the PDF
- Values extracted by regex patterns
- Values extracted by AI
- Pattern matches with surrounding context

## How It Works

### Extraction Process

The application uses a hybrid approach for extracting medical values:

1. **Pattern Matching**:

   - Uses regular expressions to find medical values based on common formats
   - Provides precise extraction for standardized reports

2. **AI-Powered Extraction**:

   - Leverages Google's Gemini AI to understand and extract values from context
   - Handles non-standard formats and more complex reports

3. **Combined Approach**:
   - Merges results from both methods
   - Uses regex findings first, falls back to AI when values are missing
   - Validates extracted values against reasonable ranges

![Extraction Process Diagram](https://raw.githubusercontent.com/username/medical-report-analyzer/main/docs/images/extraction-process.png)

### Risk Assessment

The system uses two machine learning models:

1. **Diabetes Risk Model**:

   - Support Vector Machine (SVM) classifier
   - Trained on the PIMA Diabetes dataset
   - Inputs: glucose, blood pressure, insulin, BMI, age

2. **Heart Disease Risk Model**:
   - Logistic Regression classifier
   - Trained on a standard heart disease dataset
   - Inputs: age, blood pressure, cholesterol, max heart rate, etc.

### AI Recommendations

The AI recommendation system:

- Uses extracted medical values as context
- Evaluates risk levels from the predictive models
- Generates personalized recommendations for:
  - Diet modifications
  - Exercise routines
  - Lifestyle changes
  - Follow-up medical care

## Troubleshooting

### Missing Values in Analysis

If certain medical values aren't being extracted:

1. Run the PDF info extractor:
   ```bash
   python pdf_info_extractor.py your_report.pdf
   ```
2. Check if the text is being properly extracted from the PDF
3. Examine which extraction method (regex or AI) is failing

### AI Functionality Not Working

If you're not getting AI recommendations:

1. Verify your API key is set correctly in the `.env` file
2. Run the diagnostic tool:
   ```bash
   python gemini_diagnostic.py
   ```
3. Check if your account has access to the required Gemini models

### PDF Extraction Issues

If text isn't being extracted correctly:

1. Ensure your PDF is not password-protected
2. Check if the PDF contains actual text (not just images)
3. Try the debug tool to see what text is extracted:
   ```bash
   python pdf_debug.py your_report.pdf
   ```

### Common Error Messages

| Error                             | Solution                                                     |
| --------------------------------- | ------------------------------------------------------------ |
| "Failed to extract text from PDF" | Check if the PDF is accessible and contains extractable text |
| "Google API key not found"        | Ensure you've created a .env file with your API key          |
| "Error with Gemini model"         | Run gemini_diagnostic.py to see available models             |
| "Model not loaded"                | Ensure the .pkl model files are in your project directory    |

## Contributing

Contributions to improve the Medical Report Analyzer are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for changes

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Note

This application is for educational and informational purposes only. It should not replace professional medical advice. The predictions and recommendations should be reviewed by healthcare professionals.
