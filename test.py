from medical_report_analyzer import analyze_medical_report

# Analyze a medical report
analysis = analyze_medical_report("Cheshtaa Bhardwaj Test.pdf")

# Access different aspects of the analysis
key_findings = analysis['key_findings']
medical_terms = analysis['medical_terms']
insights = analysis['insights']
precautions = analysis['precautions']
recommendations = analysis['recommendations']
print(key_findings,medical_terms,insights,precautions,recommendations)