import os
import numpy as np
import pandas as pd
import pickle
import re
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import PyPDF2
import google.generativeai as genai
from dotenv import load_dotenv
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime
import threading

# Load environment variables
load_dotenv()

# Configure Gemini API
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        print("Warning: GOOGLE_API_KEY not found in environment variables. AI analysis will be unavailable.")
    else:
        genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    GOOGLE_API_KEY = None

class MedicalReportAnalyzer:
    def __init__(self):
        self.diabetes_model = None
        self.heart_disease_model = None
        self.diabetes_scaler = None
        self.heart_scaler = None
        
        # Load or train models
        self.load_models()
        
    def load_models(self):
        """Load pre-trained models or train new ones if not available"""
        try:
            # Try to load diabetes model and scaler
            if os.path.exists('diabetes_model.pkl') and os.path.exists('diabetes_scaler.pkl'):
                self.diabetes_model = joblib.load('diabetes_model.pkl')
                self.diabetes_scaler = joblib.load('diabetes_scaler.pkl')
                print("Loaded diabetes model from file")
            else:
                print("Training diabetes model...")
                self.train_diabetes_model()
                
            # Try to load heart disease model and scaler
            if os.path.exists('heart_disease_model.pkl') and os.path.exists('heart_scaler.pkl'):
                self.heart_disease_model = joblib.load('heart_disease_model.pkl')
                self.heart_scaler = joblib.load('heart_scaler.pkl')
                print("Loaded heart disease model from file")
            else:
                print("Training heart disease model...")
                self.train_heart_disease_model()
                
        except Exception as e:
            print(f"Error loading models: {e}")
            # Fall back to training models
            self.train_diabetes_model()
            self.train_heart_disease_model()
    
    def train_diabetes_model(self):
        """Train diabetes prediction model"""
        try:
            # Load diabetes dataset
            diabetes_dataset = pd.read_csv('diabetes.csv')
            
            # Separate features and target
            X = diabetes_dataset.drop(columns='Outcome', axis=1)
            Y = diabetes_dataset['Outcome']
            
            # Standardize data
            self.diabetes_scaler = StandardScaler()
            X_standardized = self.diabetes_scaler.fit_transform(X)
            
            # Train SVM model
            self.diabetes_model = svm.SVC(kernel='linear')
            self.diabetes_model.fit(X_standardized, Y)
            
            # Save the model and scaler
            joblib.dump(self.diabetes_model, 'diabetes_model.pkl')
            joblib.dump(self.diabetes_scaler, 'diabetes_scaler.pkl')
            print("Diabetes model trained and saved")
            
        except Exception as e:
            print(f"Error training diabetes model: {e}")
            raise
    
    def train_heart_disease_model(self):
        """Train heart disease prediction model"""
        try:
            # Load heart disease dataset
            heart_data = pd.read_csv('heart_disease_data.csv')
            
            # Separate features and target
            X = heart_data.drop(columns='target', axis=1)
            Y = heart_data['target']
            
            # Standardize the data (for consistency, though LogisticRegression doesn't strictly require it)
            self.heart_scaler = StandardScaler()
            X_standardized = self.heart_scaler.fit_transform(X)
            
            # Train Logistic Regression model
            self.heart_disease_model = LogisticRegression(max_iter=1000)
            self.heart_disease_model.fit(X_standardized, Y)
            
            # Save the model and scaler
            joblib.dump(self.heart_disease_model, 'heart_disease_model.pkl')
            joblib.dump(self.heart_scaler, 'heart_scaler.pkl')
            print("Heart disease model trained and saved")
            
        except Exception as e:
            print(f"Error training heart disease model: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return None
    
    def extract_medical_values(self, text):
        """Extract relevant medical values from the report text
        Attempts to find key indicators for diabetes and heart disease
        """
        # First try extracting with regex patterns
        medical_data = self._extract_with_regex(text)
        
        # If we're missing critical values and Gemini API is available, try AI extraction
        missing_critical = (medical_data['Glucose'] is None or 
                           medical_data['BloodPressure'] is None or 
                           medical_data['Cholesterol'] is None or
                           medical_data['Age'] is None)
        
        if missing_critical and GOOGLE_API_KEY:
            ai_data = self._extract_with_gemini(text)
            # Merge AI findings with regex findings, prioritizing regex where available
            for key, value in ai_data.items():
                if medical_data.get(key) is None and value is not None:
                    medical_data[key] = value
        
        # Set default age if still not found
        if medical_data['Age'] is None:
            # Set a reasonable default adult age
            medical_data['Age'] = 45.0
            print("Note: Patient age not found in report. Using default value of 45.")
        
        return medical_data
    
    def _extract_with_regex(self, text):
        """Extract medical values using regex patterns"""
        # Dictionary to store extracted values
        medical_data = {
            # Diabetes related
            'Glucose': None,
            'BloodPressure': None,
            'Insulin': None,
            'BMI': None,
            'Age': None,
            # Heart disease related
            'Cholesterol': None,
            'RestingBP': None,
            'MaxHeartRate': None,
            'ECG': None,
        }
        
        # Regular expressions to match common medical report formats
        patterns = {
            'Glucose': r'(?:Glucose|Blood\s+glucose|Sugar)[\s:]*(\d+\.?\d*)',
            'BloodPressure': r'(?:Blood\s+pressure|BP)[\s:]*((?:\d+)/(?:\d+))',
            'Insulin': r'(?:Insulin|Serum\s+insulin)[\s:]*(\d+\.?\d*)',
            'BMI': r'(?:BMI|Body\s+Mass\s+Index)[\s:]*(\d+\.?\d*)',
            # Improved age pattern that looks for more specific patient age contexts
            'Age': r'(?:Patient\s+age|Patient\'s\s+age|Age\s+of\s+patient|Patient[:]\s+Age)[\s:]*(\d+)',
            'Cholesterol': r'(?:Cholesterol|Total\s+cholesterol)[\s:]*(\d+\.?\d*)',
            'MaxHeartRate': r'(?:Max\s+heart\s+rate|Maximum\s+heart\s+rate)[\s:]*(\d+)',
            'ECG': r'(?:ECG|EKG|Electrocardiogram)[\s:]*(\w+)',
        }
        
        # Extract values using regex
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if key == 'BloodPressure' and match.group(1):
                    # Extract systolic BP (first number) for analysis
                    systolic_bp = match.group(1).split('/')[0]
                    medical_data['BloodPressure'] = int(systolic_bp)
                elif match.group(1):
                    try:
                        # Try to convert to float, if fails keep as string
                        medical_data[key] = float(match.group(1))
                    except ValueError:
                        medical_data[key] = match.group(1)
        
        # If age wasn't found with the primary pattern, try additional patterns
        if medical_data['Age'] is None:
            age_patterns = [
                # Look for age with additional context
                r'(?:Age|Years\s+Old)[\s:]*((?:1[8-9]|[2-9][0-9]|[1-9][0-9][0-9]))\b',  # Ages 18+ followed by word boundary
                r'(?:Patient\s+is\s+|Subject\s+is\s+)(\d{1,3})[- ](?:years?|yrs?)[- ]old',  # "Patient is X years old"
                r'(?:year|yr)[- ]old[- ](\d{1,3})',  # "year-old X"
                r'(\d{1,3})[- ](?:year|yr)[- ]old',  # "X year-old"
            ]
            
            for pattern in age_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        age_value = float(match.group(1))
                        # Only accept reasonable ages (18-120)
                        if 18 <= age_value <= 120:
                            medical_data['Age'] = age_value
                            break
                    except ValueError:
                        continue
        
        # If age still not found, try DOB patterns and calculate age
        if medical_data['Age'] is None:
            dob_patterns = [
                r'(?:DOB|Date\s+of\s+Birth|Birth\s+Date)[\s:]*(\d{1,2})[/-](\d{1,2})[/-](\d{4})',  # MM/DD/YYYY or DD/MM/YYYY
                r'(?:DOB|Date\s+of\s+Birth|Birth\s+Date)[\s:]*(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY/MM/DD
            ]
            
            from datetime import datetime
            current_year = datetime.now().year
            
            for pattern in dob_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        # Try to extract year from different formats
                        if len(match.group(3)) == 4:  # If third group is year (YYYY)
                            birth_year = int(match.group(3))
                        elif len(match.group(1)) == 4:  # If first group is year (YYYY)
                            birth_year = int(match.group(1))
                        else:
                            continue
                            
                        # Calculate age - simple approximation
                        if 1900 <= birth_year <= current_year:
                            age = current_year - birth_year
                            # Only accept reasonable ages (18-120)
                            if 18 <= age <= 120:
                                medical_data['Age'] = float(age)
                                break
                    except (ValueError, IndexError):
                        continue
        
        return medical_data
    
    def _extract_with_gemini(self, text):
        """Extract medical values using Gemini AI"""
        try:
            # Start with an empty data dictionary
            medical_data = {
                'Glucose': None,
                'BloodPressure': None, 
                'Insulin': None,
                'BMI': None,
                'Age': None,
                'Cholesterol': None,
                'MaxHeartRate': None
            }
            
            # Create a prompt for Gemini to extract medical values
            prompt = f"""
            Extract the following medical values from this text from a medical report. 
            For each value, return ONLY the numeric value without units, or NULL if not found.
            If the value is a range, return the higher value.
            Format your response as a structured JSON object.
            
            Values to extract (return only these fields exactly as listed below):
            - Age: Patient's age in years
            - Glucose: Blood glucose/sugar level
            - BloodPressure: Systolic blood pressure (the first number in BP measurement)
            - Insulin: Insulin level
            - BMI: Body Mass Index
            - Cholesterol: Total cholesterol level
            - MaxHeartRate: Maximum heart rate
            
            Medical report text:
            {text[:5000]}  # Limit text to avoid token limits
            
            Response format example:
            {{
              "Age": 45,
              "Glucose": 120,
              "BloodPressure": 130,
              "Insulin": 80,
              "BMI": 28.5,
              "Cholesterol": 200,
              "MaxHeartRate": 165
            }}
            
            If a value is not found in the text, use null instead of a number.
            Respond with ONLY the JSON object, no other text.
            """
            
            # Use Gemini to extract the values
            try:
                model = genai.GenerativeModel('models/gemini-1.5-pro')
                response = model.generate_content(prompt)
                response_text = response.text
                
                # Extract JSON from response
                import json
                import re
                
                # Find JSON in the response (handling possible formatting issues)
                json_match = re.search(r'({[\s\S]*})', response_text)
                if json_match:
                    try:
                        extracted_data = json.loads(json_match.group(1))
                        
                        # Update medical_data with extracted values, converting to float where possible
                        for key in medical_data.keys():
                            if key in extracted_data and extracted_data[key] is not None:
                                try:
                                    if key == 'BloodPressure' and isinstance(extracted_data[key], str):
                                        # Handle possible "120/80" format in BP
                                        systolic = extracted_data[key].split('/')[0]
                                        medical_data[key] = float(systolic)
                                    else:
                                        medical_data[key] = float(extracted_data[key])
                                except (ValueError, TypeError):
                                    # Keep the value as is if conversion fails
                                    medical_data[key] = extracted_data[key]
                    except json.JSONDecodeError:
                        print("Failed to parse JSON from Gemini response")
            
            except Exception as e:
                print(f"Error with primary Gemini model: {e}")
                # Try alternative models if the primary one fails
                alternative_models = [
                    'models/gemini-1.5-flash',
                    'models/gemini-1.5-pro-latest',
                    'models/gemini-2.0-pro-exp'
                ]
                
                for model_name in alternative_models:
                    try:
                        model = genai.GenerativeModel(model_name)
                        response = model.generate_content(prompt)
                        # Process response as above
                        response_text = response.text
                        json_match = re.search(r'({[\s\S]*})', response_text)
                        if json_match:
                            try:
                                extracted_data = json.loads(json_match.group(1))
                                for key in medical_data.keys():
                                    if key in extracted_data and extracted_data[key] is not None:
                                        try:
                                            if key == 'BloodPressure' and isinstance(extracted_data[key], str):
                                                systolic = extracted_data[key].split('/')[0]
                                                medical_data[key] = float(systolic)
                                            else:
                                                medical_data[key] = float(extracted_data[key])
                                        except (ValueError, TypeError):
                                            medical_data[key] = extracted_data[key]
                                break  # Exit the loop if successful
                            except json.JSONDecodeError:
                                continue
                    except:
                        continue
            
            # For reasonable value checks and filtering
            if medical_data['Age'] is not None:
                # Only accept reasonable ages (18-120)
                if not (18 <= medical_data['Age'] <= 120):
                    medical_data['Age'] = None
                    
            return medical_data
            
        except Exception as e:
            print(f"Error in AI extraction: {e}")
            return {key: None for key in ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age', 'Cholesterol', 'MaxHeartRate']}
    
    def prepare_diabetes_input(self, medical_data):
        """Prepare input data for diabetes prediction model"""
        # Default values based on average population stats
        diabetes_input = [
            0,  # Pregnancies (default 0)
            medical_data.get('Glucose', 120),  # Glucose
            medical_data.get('BloodPressure', 70),  # BloodPressure
            20,  # SkinThickness (default)
            medical_data.get('Insulin', 80),  # Insulin
            medical_data.get('BMI', 25),  # BMI
            0.5,  # DiabetesPedigreeFunction (default)
            medical_data.get('Age', 45)  # Age
        ]
        
        # Convert to numpy array and reshape
        input_array = np.array(diabetes_input).reshape(1, -1)
        
        # Standardize the input
        if self.diabetes_scaler:
            standardized_input = self.diabetes_scaler.transform(input_array)
            return standardized_input
        return input_array
    
    def prepare_heart_disease_input(self, medical_data):
        """Prepare input data for heart disease prediction model"""
        # Default values based on average population stats
        heart_input = [
            medical_data.get('Age', 45),  # Age
            1,  # Sex (default: male)
            0,  # Chest pain type (default: typical angina)
            medical_data.get('BloodPressure', 120),  # Resting BP
            medical_data.get('Cholesterol', 200),  # Cholesterol
            0,  # Fasting blood sugar (default: < 120 mg/dl)
            0,  # Resting ECG (default: normal)
            medical_data.get('MaxHeartRate', 150),  # Max heart rate
            0,  # Exercise induced angina (default: no)
            0,  # ST depression (default)
            1,  # Slope of peak exercise ST segment (default: upsloping)
            0,  # Number of major vessels (default)
            1,  # Thal (default: normal)
        ]
        
        # Convert to numpy array and reshape
        input_array = np.array(heart_input).reshape(1, -1)
        
        # Standardize the input
        if self.heart_scaler:
            standardized_input = self.heart_scaler.transform(input_array)
            return standardized_input
        return input_array
    
    def predict_diabetes(self, medical_data):
        """Predict diabetes risk based on medical data"""
        if not self.diabetes_model:
            return {"error": "Diabetes model not loaded"}
        
        try:
            input_data = self.prepare_diabetes_input(medical_data)
            prediction = self.diabetes_model.predict(input_data)
            probability = None
            
            # Try to get probability
            try:
                probability = self.diabetes_model.predict_proba(input_data)
            except:
                pass
            
            result = {
                "has_diabetes": bool(prediction[0]),
                "probability": probability[0][1] if probability is not None else None
            }
            
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def predict_heart_disease(self, medical_data):
        """Predict heart disease risk based on medical data"""
        if not self.heart_disease_model:
            return {"error": "Heart disease model not loaded"}
        
        try:
            input_data = self.prepare_heart_disease_input(medical_data)
            prediction = self.heart_disease_model.predict(input_data)
            probability = None
            
            # Try to get probability
            try:
                probability = self.heart_disease_model.predict_proba(input_data)
            except:
                pass
            
            result = {
                "has_heart_disease": bool(prediction[0]),
                "probability": probability[0][1] if probability is not None else None
            }
            
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def generate_ai_report(self, medical_data, diabetes_result, heart_result):
        """Generate AI-powered health report using Gemini AI"""
        if not GOOGLE_API_KEY:
            return "AI analysis unavailable. Please set up your GOOGLE_API_KEY."
        
        try:
            # Prepare the prompt
            age = medical_data.get('Age', 'unknown')
            glucose = medical_data.get('Glucose', 'unknown')
            bp = medical_data.get('BloodPressure', 'unknown')
            bmi = medical_data.get('BMI', 'unknown')
            cholesterol = medical_data.get('Cholesterol', 'unknown')
            
            diabetes_risk = "high" if diabetes_result.get('has_diabetes', False) else "low"
            heart_risk = "high" if heart_result.get('has_heart_disease', False) else "low"
            
            prompt = f"""
            Based on a patient's medical report, provide a detailed health analysis and recommendations.
            
            Patient Data:
            - Age: {age}
            - Glucose level: {glucose} mg/dL
            - Blood Pressure: {bp} mmHg
            - BMI: {bmi}
            - Cholesterol: {cholesterol} mg/dL
            
            Risk Assessment:
            - Diabetes Risk: {diabetes_risk}
            - Heart Disease Risk: {heart_risk}
            
            Please provide the following in your analysis:
            1. A brief explanation of what these values mean for the patient's health
            2. Dietary recommendations (specific foods to eat and avoid)
            3. Exercise recommendations (types, frequency, duration)
            4. Lifestyle changes that could improve their health
            5. Precautions they should take given their risk factors
            6. Any vitamins or supplements that might be beneficial
            7. When they should follow up with a doctor
            
            Format your response in clear sections with headers.
            """
            
            # Use a model from the available models list (from diagnostic output)
            try:
                model = genai.GenerativeModel('models/gemini-1.5-pro')
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                # If the first model fails, try alternatives
                alternative_models = [
                    'models/gemini-1.5-flash',
                    'models/gemini-1.5-pro-latest',
                    'models/gemini-2.0-pro-exp'
                ]
                
                for model_name in alternative_models:
                    try:
                        model = genai.GenerativeModel(model_name)
                        response = model.generate_content(prompt)
                        return response.text
                    except:
                        continue
                
                # If all attempts fail, return error message
                return f"Error generating AI report: {e}\n\nPlease check your API key and run 'python run.py diagnostic' to see available models."
                
        except Exception as e:
            return f"Error generating AI report: {e}\n\nPlease check your API key or try again later."
    
    def analyze_report(self, pdf_path):
        """Analyze medical report PDF and return comprehensive results"""
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return {"error": "Failed to extract text from PDF"}
        
        # Extract medical values from text
        medical_data = self.extract_medical_values(text)
        
        # Make predictions
        diabetes_result = self.predict_diabetes(medical_data)
        heart_result = self.predict_heart_disease(medical_data)
        
        # Generate AI report
        ai_report = self.generate_ai_report(medical_data, diabetes_result, heart_result)
        
        # Compile final results
        results = {
            "extracted_data": medical_data,
            "diabetes_analysis": diabetes_result,
            "heart_disease_analysis": heart_result,
            "ai_report": ai_report,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return results

class MedicalReportGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Report Analyzer")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Initialize analyzer
        self.analyzer = MedicalReportAnalyzer()
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Medical Report Analyzer", font=("Helvetica", 18, "bold"))
        title_label.pack(pady=10)
        
        # File selection frame
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=10)
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=60)
        file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        browse_button = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_button.pack(side=tk.LEFT, padx=5)
        
        analyze_button = ttk.Button(file_frame, text="Analyze Report", command=self.analyze_report)
        analyze_button.pack(side=tk.LEFT, padx=5)
        
        # Progress indicator
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, 
                                        length=100, mode='indeterminate', 
                                        variable=self.progress_var)
        self.progress.pack(fill=tk.X, pady=10)
        
        # Results frame with tabs
        self.tab_control = ttk.Notebook(main_frame)
        self.tab_control.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create tabs
        self.summary_tab = ttk.Frame(self.tab_control)
        self.ai_report_tab = ttk.Frame(self.tab_control)
        self.extracted_data_tab = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.summary_tab, text="Summary")
        self.tab_control.add(self.ai_report_tab, text="AI Recommendations")
        self.tab_control.add(self.extracted_data_tab, text="Extracted Data")
        
        # Summary tab content
        self.summary_frame = ttk.Frame(self.summary_tab, padding=10)
        self.summary_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initial summary text
        self.summary_text = tk.Text(self.summary_frame, wrap=tk.WORD, height=20)
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        self.summary_text.insert(tk.END, "Upload a medical report (PDF) and click 'Analyze Report' to begin.")
        self.summary_text.config(state=tk.DISABLED)
        
        # AI report tab content
        self.ai_report_frame = ttk.Frame(self.ai_report_tab, padding=10)
        self.ai_report_frame.pack(fill=tk.BOTH, expand=True)
        
        self.ai_report_text = tk.Text(self.ai_report_frame, wrap=tk.WORD, height=20)
        self.ai_report_text.pack(fill=tk.BOTH, expand=True)
        self.ai_report_text.config(state=tk.DISABLED)
        
        # Extracted data tab content
        self.extracted_data_frame = ttk.Frame(self.extracted_data_tab, padding=10)
        self.extracted_data_frame.pack(fill=tk.BOTH, expand=True)
        
        self.extracted_data_text = tk.Text(self.extracted_data_frame, wrap=tk.WORD, height=20)
        self.extracted_data_text.pack(fill=tk.BOTH, expand=True)
        self.extracted_data_text.config(state=tk.DISABLED)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
    
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Medical Report",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
    
    def analyze_report(self):
        file_path = self.file_path_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid PDF file")
            return
        
        # Clear previous results
        self.clear_results()
        
        # Update status
        self.status_var.set("Analyzing report...")
        self.progress.start()
        
        # Run analysis in a separate thread to keep UI responsive
        threading.Thread(target=self._analyze_report_thread, args=(file_path,), daemon=True).start()
    
    def _analyze_report_thread(self, file_path):
        try:
            # Perform analysis
            results = self.analyzer.analyze_report(file_path)
            
            # Update UI with results
            self.root.after(0, lambda: self.update_results(results))
        except Exception as e:
            self.root.after(0, lambda: self.show_error(f"Analysis failed: {str(e)}"))
    
    def update_results(self, results):
        if "error" in results:
            self.show_error(results["error"])
            return
        
        # Update summary tab
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        
        diabetes_result = results["diabetes_analysis"]
        heart_result = results["heart_disease_analysis"]
        
        summary = f"""ANALYSIS SUMMARY
        
Report analyzed on: {results['timestamp']}

DIABETES RISK ASSESSMENT:
Risk: {"High" if diabetes_result.get('has_diabetes', False) else "Low"}
{"Probability: " + str(round(diabetes_result.get('probability', 0) * 100, 2)) + "%" if diabetes_result.get('probability') is not None else ""}

HEART DISEASE RISK ASSESSMENT:
Risk: {"High" if heart_result.get('has_heart_disease', False) else "Low"}
{"Probability: " + str(round(heart_result.get('probability', 0) * 100, 2)) + "%" if heart_result.get('probability') is not None else ""}

RECOMMENDATIONS:
Please see the AI Recommendations tab for detailed advice on diet, exercise, 
and lifestyle recommendations based on your results.
"""
        
        self.summary_text.insert(tk.END, summary)
        self.summary_text.config(state=tk.DISABLED)
        
        # Update AI report tab
        self.ai_report_text.config(state=tk.NORMAL)
        self.ai_report_text.delete(1.0, tk.END)
        self.ai_report_text.insert(tk.END, results["ai_report"])
        self.ai_report_text.config(state=tk.DISABLED)
        
        # Update extracted data tab
        self.extracted_data_text.config(state=tk.NORMAL)
        self.extracted_data_text.delete(1.0, tk.END)
        
        extracted_data = results["extracted_data"]
        data_text = "EXTRACTED MEDICAL DATA:\n\n"
        for key, value in extracted_data.items():
            data_text += f"{key}: {value}\n"
        
        self.extracted_data_text.insert(tk.END, data_text)
        self.extracted_data_text.config(state=tk.DISABLED)
        
        # Update status
        self.status_var.set("Analysis complete")
        self.progress.stop()
    
    def show_error(self, message):
        self.status_var.set("Error")
        self.progress.stop()
        messagebox.showerror("Error", message)
    
    def clear_results(self):
        # Clear all result displays
        for text_widget in [self.summary_text, self.ai_report_text, self.extracted_data_text]:
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            text_widget.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = MedicalReportGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 