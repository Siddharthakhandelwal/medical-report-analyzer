# Medical Report Analyzer

A Python-based tool for extracting and analyzing medical reports from PDF files. This tool helps in understanding medical reports by providing structured insights, identifying key medical terms, and offering relevant precautions and recommendations.

## Features

- PDF text extraction
- Medical term identification and categorization
- Medical abbreviation expansion
- Condition-specific analysis
- Symptom tracking
- Precautions and recommendations generation
- Detailed medical insights

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd medical-report-analyzer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Project Structure

```
medical-report-analyzer/
├── pdf_text_extractor.py    # PDF text extraction module
├── medical_report_analyzer.py  # Main analysis module
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Usage Guide

### Step 1: Prepare Your Medical Report
- Ensure your medical report is in PDF format
- Make sure the PDF is not password-protected
- The PDF should contain readable text (not scanned images)

### Step 2: Run the Analyzer
There are two ways to use the analyzer:

#### Method 1: Command Line
```bash
python medical_report_analyzer.py
```
When prompted, enter the path to your PDF file.

#### Method 2: Import as Module
```python
from medical_report_analyzer import analyze_medical_report

# Analyze a medical report
analysis = analyze_medical_report("path/to/your/medical_report.pdf")

# Access different aspects of the analysis
print(analysis['detailed_analysis']['analysis'])
print(analysis['precautions'])
print(analysis['recommendations'])
```

### Step 3: Understanding the Output

The analyzer provides several types of information:

1. **Key Findings**
   - Important medical information from the report
   - Diagnosis and test results

2. **Medical Terms**
   - Categorized medical terminology
   - Context of each term's usage

3. **Medical Abbreviations**
   - Explanation of medical abbreviations
   - Full forms of abbreviated terms

4. **Detailed Analysis**
   - Comprehensive analysis of identified conditions
   - Description of conditions
   - Associated symptoms
   - Treatment information

5. **Precautions**
   - General health precautions
   - Condition-specific precautions
   - Medication-related precautions

6. **Recommendations**
   - General health recommendations
   - Treatment follow-up suggestions
   - Lifestyle advice

## Example Output

```
Medical Report Analysis:

Key Findings:
- Blood pressure: 140/90 mmHg
- Heart rate: 72 bpm
- Cholesterol levels: Elevated

Medical Terms:
Conditions:
- Hypertension (Context: Patient diagnosed with hypertension)
- Hypercholesterolemia (Context: Elevated cholesterol levels noted)

Symptoms:
- Headache (Context: Patient reports occasional headaches)
- Fatigue (Context: Experiencing increased fatigue)

Detailed Analysis:
Identified Conditions:
Hypertension:
Description: High blood pressure
Symptoms: headache, dizziness, blurred vision
Precautions: reduce salt intake, exercise regularly, monitor blood pressure
Treatments: medication, lifestyle changes

General Health Recommendations:
1. Maintain a balanced diet
2. Exercise regularly
3. Get adequate sleep
4. Stay hydrated
5. Schedule regular check-ups
```

## Customization

### Adding New Medical Conditions

To add new medical conditions to the analyzer's knowledge base, modify the `medical_knowledge` dictionary in `medical_report_analyzer.py`:

```python
self.medical_knowledge = {
    'new_condition': {
        'description': 'Description of the condition',
        'symptoms': ['symptom1', 'symptom2'],
        'precautions': ['precaution1', 'precaution2'],
        'treatments': ['treatment1', 'treatment2']
    }
}
```

### Adding New Medical Terms

To add new medical terms or categories, modify the `medical_keywords` dictionary:

```python
self.medical_keywords = {
    'new_category': ['term1', 'term2', 'term3']
}
```

## Troubleshooting

1. **PDF Text Extraction Issues**
   - Ensure the PDF contains selectable text
   - Try converting the PDF to a newer format
   - Check if the PDF is password-protected

2. **Installation Issues**
   - Make sure you're using Python 3.7 or higher
   - Try installing packages individually if requirements.txt fails
   - Check if all dependencies are properly installed

3. **Analysis Issues**
   - Verify that the medical report contains clear, readable text
   - Check if medical terms are properly spelled
   - Ensure the report is in English

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## Acknowledgments

- spaCy for natural language processing
- PyPDF2 for PDF text extraction
- NLTK for text processing 