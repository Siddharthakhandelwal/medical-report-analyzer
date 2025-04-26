#!/usr/bin/env python3
"""
Generate Sample Medical Report

This script generates sample medical report PDFs with customizable values
for testing the Medical Report Analyzer system.

Usage:
    python generate_sample_report.py [options]

Example:
    python generate_sample_report.py --glucose 130 --cholesterol 210 --output custom_report.pdf
"""

import os
import sys
import argparse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate sample medical report PDF')
    
    # Output file
    parser.add_argument('--output', '-o', default='sample_medical_report.pdf',
                        help='Output PDF file name (default: sample_medical_report.pdf)')
    
    # Patient information
    parser.add_argument('--name', default='John Doe',
                        help='Patient name (default: John Doe)')
    parser.add_argument('--age', type=int, default=55,
                        help='Patient age (default: 55)')
    parser.add_argument('--id', default='P12345',
                        help='Patient ID (default: P12345)')
    
    # Medical values - Diabetes related
    parser.add_argument('--glucose', type=float, default=140,
                        help='Glucose level in mg/dL (default: 140)')
    parser.add_argument('--bp', default='135/85',
                        help='Blood pressure in mmHg (default: 135/85)')
    parser.add_argument('--insulin', type=float, default=95,
                        help='Insulin level in uIU/mL (default: 95)')
    parser.add_argument('--bmi', type=float, default=28.5,
                        help='BMI value (default: 28.5)')
    
    # Medical values - Heart related
    parser.add_argument('--cholesterol', type=float, default=220,
                        help='Total cholesterol in mg/dL (default: 220)')
    parser.add_argument('--hdl', type=float, default=45,
                        help='HDL cholesterol in mg/dL (default: 45)')
    parser.add_argument('--ldl', type=float, default=155,
                        help='LDL cholesterol in mg/dL (default: 155)')
    parser.add_argument('--max-heart-rate', type=int, default=165,
                        help='Maximum heart rate in bpm (default: 165)')
    
    return parser.parse_args()

def create_medical_report(args):
    """Create a sample medical report PDF with the specified values"""
    print(f"Creating sample medical report: {args.output}")
    
    c = canvas.Canvas(args.output, pagesize=letter)
    width, height = letter
    
    # Add a title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(30, height - 50, "Medical Laboratory Report")
    
    # Add a patient section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, height - 90, "Patient Information:")
    c.setFont("Helvetica", 10)
    c.drawString(30, height - 110, f"Name: {args.name}")
    c.drawString(30, height - 125, f"Age: {args.age}")
    c.drawString(30, height - 140, "Date of Birth: 01/15/1968")  # Fixed value for simplicity
    c.drawString(30, height - 155, f"Patient ID: {args.id}")
    
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
    
    # Prepare test data with user-provided values
    test_data = [
        ["Glucose", str(args.glucose), "70-99", "mg/dL"],
        ["Hemoglobin A1c", "6.2", "4.0-5.6", "%"],
        ["Total Cholesterol", str(args.cholesterol), "<200", "mg/dL"],
        ["HDL Cholesterol", str(args.hdl), ">40", "mg/dL"],
        ["LDL Cholesterol", str(args.ldl), "<100", "mg/dL"],
        ["Triglycerides", "180", "<150", "mg/dL"],
        ["Blood Pressure", args.bp, "<120/80", "mmHg"],
        ["BMI", str(args.bmi), "18.5-24.9", "kg/m²"],
        ["Insulin", str(args.insulin), "3-25", "uIU/mL"],
        ["Max Heart Rate", str(args.max_heart_rate), "100-170", "bpm"]
    ]
    
    # Add test values to PDF
    y_position = height - 245
    for test in test_data:
        c.drawString(30, y_position, test[0])
        c.drawString(200, y_position, test[1])
        c.drawString(300, y_position, test[2])
        c.drawString(400, y_position, test[3])
        y_position -= 20
    
    # Add a risk assessment section
    c.setFont("Helvetica-Bold", 12)
    y_position -= 20
    c.drawString(30, y_position, "Risk Assessment:")
    c.setFont("Helvetica", 10)
    
    # Calculate glucose risk
    glucose_risk = "High" if args.glucose > 125 else "Normal"
    cholesterol_risk = "High" if args.cholesterol > 200 else "Normal"
    
    y_position -= 20
    c.drawString(30, y_position, f"Diabetes Risk: {glucose_risk}")
    y_position -= 20
    c.drawString(30, y_position, f"Heart Disease Risk: {cholesterol_risk}")
    
    # Add a note at the bottom
    c.setFont("Helvetica-Italic", 9)
    c.drawString(30, 50, "This is a sample report generated for software testing purposes only. Not for medical use.")
    
    # Save the PDF
    c.save()
    print(f"Report generated successfully: {args.output}")
    return args.output

def main():
    """Main entry point for sample report generator"""
    args = parse_arguments()
    
    try:
        # Create the report
        output_file = create_medical_report(args)
        
        # Suggest next steps
        print("\nYou can now analyze this report using:")
        print(f"  python run.py cli {output_file}")
        print("  or")
        print("  python run.py gui")
        print("  (then browse to select the file)")
        
        return 0
    
    except Exception as e:
        print(f"Error creating report: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 