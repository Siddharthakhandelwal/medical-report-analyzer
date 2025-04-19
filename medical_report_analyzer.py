import spacy
import nltk
from nltk.tokenize import sent_tokenize
from typing import Dict, List, Tuple
import re
from collections import defaultdict
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class MedicalReportAnalyzer:
    def __init__(self):
        # Load the English language model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Medical keywords and their categories
        self.medical_keywords = {
            'conditions': ['diagnosis', 'condition', 'disease', 'illness', 'infection', 'disorder', 'syndrome'],
            'symptoms': ['symptom', 'pain', 'fever', 'cough', 'headache', 'fatigue', 'nausea', 'vomiting', 'dizziness'],
            'medications': ['medication', 'drug', 'prescription', 'dose', 'mg', 'tablet', 'capsule', 'injection'],
            'tests': ['test', 'scan', 'x-ray', 'mri', 'ct', 'blood test', 'urine test', 'ultrasound'],
            'vitals': ['blood pressure', 'heart rate', 'temperature', 'pulse', 'respiratory rate', 'oxygen saturation'],
            'treatments': ['treatment', 'therapy', 'surgery', 'procedure', 'operation', 'rehabilitation']
        }
        
        # Common medical abbreviations
        self.medical_abbreviations = {
            'BP': 'Blood Pressure',
            'HR': 'Heart Rate',
            'RR': 'Respiratory Rate',
            'CBC': 'Complete Blood Count',
            'MRI': 'Magnetic Resonance Imaging',
            'CT': 'Computed Tomography',
            'ECG': 'Electrocardiogram',
            'CXR': 'Chest X-Ray',
            'CMP': 'Comprehensive Metabolic Panel',
            'LFT': 'Liver Function Test',
            'RFT': 'Renal Function Test'
        }
        
        # Medical knowledge base (simplified for example)
        self.medical_knowledge = {
            'hypertension': {
                'description': 'High blood pressure',
                'symptoms': ['headache', 'dizziness', 'blurred vision'],
                'precautions': ['reduce salt intake', 'exercise regularly', 'monitor blood pressure'],
                'treatments': ['medication', 'lifestyle changes']
            },
            'diabetes': {
                'description': 'High blood sugar levels',
                'symptoms': ['increased thirst', 'frequent urination', 'fatigue'],
                'precautions': ['monitor blood sugar', 'maintain healthy diet', 'regular exercise'],
                'treatments': ['insulin', 'oral medications', 'diet control']
            }
            # Add more conditions as needed
        }

    def analyze_report(self, extracted_text: Dict[str, List[str]]) -> Dict:
        """
        Analyze the medical report and provide detailed insights.
        
        Args:
            extracted_text (Dict): Dictionary containing structured text from PDF
            
        Returns:
            Dict: Analysis results including findings, insights, and recommendations
        """
        # Combine all text for analysis
        full_text = ' '.join(extracted_text['pages'])
        
        # Initialize results dictionary
        analysis = {
            'key_findings': [],
            'medical_terms': defaultdict(list),
            'abbreviations': [],
            'insights': [],
            'precautions': [],
            'recommendations': [],
            'detailed_analysis': {}
        }
        
        # Process the text with spaCy
        doc = self.nlp(full_text)
        
        # Extract medical terms and categorize them
        for token in doc:
            for category, keywords in self.medical_keywords.items():
                if token.text.lower() in keywords:
                    context = self._get_context(token, doc)
                    analysis['medical_terms'][category].append({
                        'term': token.text,
                        'context': context
                    })
        
        # Find medical abbreviations
        for abbr, full_form in self.medical_abbreviations.items():
            if abbr in full_text:
                analysis['abbreviations'].append({
                    'abbreviation': abbr,
                    'full_form': full_form
                })
        
        # Extract key findings
        analysis['key_findings'] = self._extract_key_findings(doc)
        
        # Generate detailed analysis using free medical resources
        analysis['detailed_analysis'] = self._generate_detailed_analysis(full_text)
        
        # Generate insights and recommendations
        analysis['insights'] = self._generate_insights(analysis)
        analysis['precautions'] = self._generate_precautions(analysis)
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis

    def _get_context(self, token, doc, window_size=5) -> str:
        """Get context around a token"""
        start = max(0, token.i - window_size)
        end = min(len(doc), token.i + window_size + 1)
        return ' '.join([t.text for t in doc[start:end]])

    def _extract_key_findings(self, doc) -> List[str]:
        """Extract key findings from the document"""
        findings = []
        for sent in doc.sents:
            if any(keyword in sent.text.lower() for keyword in ['findings', 'results', 'diagnosis', 'conclusion']):
                findings.append(sent.text.strip())
        return findings

    def _search_medical_knowledge(self, term: str) -> Dict:
        """Search for medical information in the knowledge base"""
        term = term.lower()
        for condition, info in self.medical_knowledge.items():
            if condition in term or term in condition:
                return info
        return {}

    def _generate_detailed_analysis(self, text: str) -> Dict:
        """Generate detailed medical analysis using free resources"""
        try:
            # Extract potential conditions from the text
            conditions = []
            for token in self.nlp(text):
                if token.text.lower() in self.medical_knowledge:
                    conditions.append(token.text.lower())
            
            # Generate analysis based on identified conditions
            analysis_text = "Medical Report Analysis:\n\n"
            
            if conditions:
                analysis_text += "Identified Conditions:\n"
                for condition in conditions:
                    info = self.medical_knowledge[condition]
                    analysis_text += f"\n{condition.capitalize()}:\n"
                    analysis_text += f"Description: {info['description']}\n"
                    analysis_text += f"Symptoms: {', '.join(info['symptoms'])}\n"
                    analysis_text += f"Precautions: {', '.join(info['precautions'])}\n"
                    analysis_text += f"Treatments: {', '.join(info['treatments'])}\n"
            else:
                analysis_text += "No specific medical conditions identified in the report.\n"
            
            # Add general health recommendations
            analysis_text += "\nGeneral Health Recommendations:\n"
            analysis_text += "1. Maintain a balanced diet\n"
            analysis_text += "2. Exercise regularly\n"
            analysis_text += "3. Get adequate sleep\n"
            analysis_text += "4. Stay hydrated\n"
            analysis_text += "5. Schedule regular check-ups\n"
            
            return {
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis_text
            }
            
        except Exception as e:
            print(f"Error in detailed analysis: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'analysis': "Unable to generate detailed analysis due to an error."
            }

    def _generate_insights(self, analysis: Dict) -> List[str]:
        """Generate insights based on the analysis"""
        insights = []
        
        # Add insights from detailed analysis
        if 'detailed_analysis' in analysis and 'analysis' in analysis['detailed_analysis']:
            insights.append(analysis['detailed_analysis']['analysis'])
        
        # Add insights from medical terms
        if analysis['medical_terms']['conditions']:
            conditions = [term['term'] for term in analysis['medical_terms']['conditions']]
            insights.append(f"Identified medical conditions: {', '.join(conditions)}")
        
        if analysis['medical_terms']['symptoms']:
            symptoms = [term['term'] for term in analysis['medical_terms']['symptoms']]
            insights.append(f"Reported symptoms: {', '.join(symptoms)}")
        
        if analysis['medical_terms']['medications']:
            medications = [term['term'] for term in analysis['medical_terms']['medications']]
            insights.append(f"Prescribed medications: {', '.join(medications)}")
        
        return insights

    def _generate_precautions(self, analysis: Dict) -> List[str]:
        """Generate precautions based on the analysis"""
        precautions = []
        
        # Add precautions from detailed analysis
        if 'detailed_analysis' in analysis and 'analysis' in analysis['detailed_analysis']:
            analysis_text = analysis['detailed_analysis']['analysis']
            if "Precautions:" in analysis_text:
                precautions_section = analysis_text.split("Precautions:")[1].split("\n\n")[0]
                precautions.extend([p.strip() for p in precautions_section.split("\n") if p.strip()])
        
        # Add general precautions
        precautions.append("Follow prescribed medication schedule strictly")
        precautions.append("Maintain regular follow-up appointments")
        precautions.append("Monitor symptoms and report any changes")
        
        # Add condition-specific precautions
        if any('pain' in term['term'].lower() for term in analysis['medical_terms']['symptoms']):
            precautions.append("Avoid strenuous activities that may exacerbate pain")
        
        if any('fever' in term['term'].lower() for term in analysis['medical_terms']['symptoms']):
            precautions.append("Monitor temperature regularly")
            precautions.append("Stay hydrated and rest adequately")
        
        return precautions

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on the analysis"""
        recommendations = []
        
        # Add recommendations from detailed analysis
        if 'detailed_analysis' in analysis and 'analysis' in analysis['detailed_analysis']:
            analysis_text = analysis['detailed_analysis']['analysis']
            if "General Health Recommendations:" in analysis_text:
                rec_section = analysis_text.split("General Health Recommendations:")[1].split("\n\n")[0]
                recommendations.extend([r.strip() for r in rec_section.split("\n") if r.strip()])
        
        # Add general recommendations
        recommendations.append("Schedule regular check-ups with your healthcare provider")
        recommendations.append("Maintain a healthy lifestyle with proper diet and exercise")
        
        # Add test-specific recommendations
        if analysis['medical_terms']['tests']:
            recommendations.append("Follow up on all recommended tests and procedures")
        
        # Add medication-specific recommendations
        if analysis['medical_terms']['medications']:
            recommendations.append("Keep track of medication side effects")
            recommendations.append("Do not stop medications without consulting your doctor")
        
        return recommendations

def analyze_medical_report(pdf_path: str) -> Dict:
    """
    Main function to analyze a medical report from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        Dict: Complete analysis of the medical report
    """
    from pdf_text_extractor import extract_text_from_pdf
    
    # Extract text from PDF
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # Initialize analyzer
    analyzer = MedicalReportAnalyzer()
    
    # Analyze the report
    analysis = analyzer.analyze_report(extracted_text)
    
    return analysis

# Example usage
if __name__ == "__main__":

    pdf_path = input("Enter a pdf path: ")#"sterling-accuris-pathology-sample-report-unlocked.pdf"  # Replace with your PDF file path
    analysis = analyze_medical_report(pdf_path)
    
    # Print the analysis results
    print("\nMedical Report Analysis:")
    print("\nKey Findings:")
    for finding in analysis['key_findings']:
        print(f"- {finding}")
    
    print("\nMedical Terms:")
    for category, terms in analysis['medical_terms'].items():
        print(f"\n{category.capitalize()}:")
        for term in terms:
            print(f"- {term['term']} (Context: {term['context']})")
    
    print("\nMedical Abbreviations:")
    for abbr in analysis['abbreviations']:
        print(f"- {abbr['abbreviation']}: {abbr['full_form']}")
    
    print("\nDetailed Analysis:")
    print(analysis['detailed_analysis']['analysis'])
    
    print("\nPrecautions:")
    for precaution in analysis['precautions']:
        print(f"- {precaution}")
    
    print("\nRecommendations:")
    for recommendation in analysis['recommendations']:
        print(f"- {recommendation}") 