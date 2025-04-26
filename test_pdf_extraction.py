"""
Test PDF Extraction for Medical Report Analyzer

This script tests the PDF extraction functionality by:
1. Creating a sample medical report PDF with test data
2. Extracting the data using the analyzer
3. Verifying the extraction works correctly

Requires: reportlab for PDF generation, PyPDF2 for extraction
"""

import os
import argparse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import re
from medical_report_analyzer import MedicalReportAnalyzer

def create_test_pdf(filename="test_medical_report.pdf"):
    """Create a sample medical report PDF for testing"""
    print(f"Creating test medical report PDF: {filename}")
    
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Add a title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(30, height - 50, "Medical Laboratory Report")
    
    # Add a patient section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, height - 90, "Patient Information:")
    c.setFont("Helvetica", 10)
    c.drawString(30, height - 110, "Name: John Doe")
    c.drawString(30, height - 125, "Age: 55")
    c.drawString(30, height - 140, "Date of Birth: 01/15/1968")
    c.drawString(30, height - 155, "Patient ID: P12345")
    
    # Add blood test results
    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, height - 195, "Blood Test Results:")
    c.setFont("Helvetica", 10)
    
    # Table header
    c.drawString(30, height - 215, "Test")
    c.drawString(200, height - 215, "Result")
    c.drawString(300, height - 215, "Normal Range")
    c.drawString(400, height - 215, "Units")
    
    # Draw a line
    c.line(30, height - 225, 550, height - 225)
    
    # Test values - these should be extractable by the analyzer
    test_data = [
        ["Glucose", "140", "70-99", "mg/dL"],
        ["Hemoglobin A1c", "6.2", "4.0-5.6", "%"],
        ["Total Cholesterol", "220", "<200", "mg/dL"],
        ["HDL Cholesterol", "45", ">40", "mg/dL"],
        ["LDL Cholesterol", "155", "<100", "mg/dL"],
        ["Triglycerides", "180", "<150", "mg/dL"],
        ["Blood Pressure", "135/85", "<120/80", "mmHg"],
        ["BMI", "28.5", "18.5-24.9", "kg/m²"],
        ["Insulin", "95", "3-25", "uIU/mL"],
        ["Max Heart Rate", "165", "100-170", "bpm"]
    ]
    
    # Add test values to PDF
    y_position = height - 245
    for test in test_data:
        c.drawString(30, y_position, test[0])
        c.drawString(200, y_position, test[1])
        c.drawString(300, y_position, test[2])
        c.drawString(400, y_position, test[3])
        y_position -= 20
    
    # Add a note at the bottom
    c.setFont("Helvetica-Italic", 9)
    c.drawString(30, 50, "This is a test report generated for software testing purposes only. Not for medical use.")
    
    # Save the PDF
    c.save()
    print(f"Test PDF created successfully: {filename}")
    return filename

def test_extraction(remove_file=False):
    """Test the extraction functionality of the MedicalReportAnalyzer"""
    # Create the test PDF
    test_pdf = create_test_pdf()
    
    # Initialize the analyzer
    analyzer = MedicalReportAnalyzer()
    
    # Extract text from the PDF
    print("Extracting text from PDF...")
    text = analyzer.extract_text_from_pdf(test_pdf)
    
    if not text:
        print("Error: Failed to extract text from PDF")
        return False
    
    print("\nExtracted text sample:")
    print(text[:200] + "...")  # Print first 200 chars
    
    # Extract medical values
    print("\nExtracting medical values...")
    medical_data = analyzer.extract_medical_values(text)
    
    # Display extracted values
    print("\nExtracted medical values:")
    for key, value in medical_data.items():
        if value is not None:
            print(f"  {key}: {value}")
    
    # Verify key values were extracted correctly
    expected_values = {
        'Glucose': 140.0,
        'BloodPressure': 135,
        'Insulin': 95.0,
        'BMI': 28.5,
        'Age': 55,
        'Cholesterol': 220.0,
        'MaxHeartRate': 165.0
    }
    
    # Check if values match expected values
    print("\nVerifying extraction accuracy:")
    success = True
    for key, expected in expected_values.items():
        actual = medical_data.get(key)
        if actual == expected:
            print(f"  ✓ {key}: {actual} (matches expected value)")
        else:
            print(f"  ✗ {key}: {actual} (expected {expected})")
            success = False
    
    # Clean up - remove test PDF if requested
    if remove_file:
        try:
            os.remove(test_pdf)
            print(f"\nTest PDF {test_pdf} removed")
        except:
            print(f"\nCould not remove test PDF {test_pdf}")
    else:
        print(f"\nTest PDF {test_pdf} saved for further testing")
    
    if success:
        print("\nExtraction test PASSED!")
    else:
        print("\nExtraction test FAILED: Some values were not extracted correctly")
    
    return success

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test PDF extraction functionality')
    parser.add_argument('--remove', action='store_true', help='Remove the test PDF after testing')
    
    return parser.parse_args()

if __name__ == "__main__":
    try:
        print("Testing PDF extraction functionality...")
        args = parse_arguments()
        test_extraction(remove_file=args.remove)
    except Exception as e:
        print(f"Error during testing: {e}")
        print("\nNote: This test requires reportlab and PyPDF2 packages.")
        print("Install them with: pip install reportlab PyPDF2") 