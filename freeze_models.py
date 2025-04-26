"""
Freeze Models for Medical Report Analyzer

This script trains and saves the diabetes and heart disease prediction models
for faster loading during inference. Run this once to generate the model files.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_and_save_diabetes_model():
    """Train and save the diabetes prediction model"""
    print("Training diabetes model...")
    
    try:
        # Load diabetes dataset
        diabetes_dataset = pd.read_csv('diabetes.csv')
        
        # Separate features and target
        X = diabetes_dataset.drop(columns='Outcome', axis=1)
        Y = diabetes_dataset['Outcome']
        
        # Standardize data
        scaler = StandardScaler()
        X_standardized = scaler.fit_transform(X)
        
        # Train SVM model
        model = svm.SVC(kernel='linear', probability=True)
        model.fit(X_standardized, Y)
        
        # Test model accuracy
        X_train, X_test, Y_train, Y_test = train_test_split(X_standardized, Y, test_size=0.2, random_state=2)
        accuracy = model.score(X_test, Y_test)
        print(f"Diabetes model accuracy: {accuracy:.4f}")
        
        # Save the model and scaler
        joblib.dump(model, 'diabetes_model.pkl')
        joblib.dump(scaler, 'diabetes_scaler.pkl')
        print("Diabetes model saved to 'diabetes_model.pkl'")
        print("Diabetes scaler saved to 'diabetes_scaler.pkl'")
        
    except Exception as e:
        print(f"Error training diabetes model: {e}")
        raise

def train_and_save_heart_disease_model():
    """Train and save the heart disease prediction model"""
    print("Training heart disease model...")
    
    try:
        # Load heart disease dataset
        heart_data = pd.read_csv('heart_disease_data.csv')
        
        # Separate features and target
        X = heart_data.drop(columns='target', axis=1)
        Y = heart_data['target']
        
        # Standardize data
        scaler = StandardScaler()
        X_standardized = scaler.fit_transform(X)
        
        # Train Logistic Regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_standardized, Y)
        
        # Test model accuracy
        X_train, X_test, Y_train, Y_test = train_test_split(X_standardized, Y, test_size=0.2, random_state=2)
        accuracy = model.score(X_test, Y_test)
        print(f"Heart disease model accuracy: {accuracy:.4f}")
        
        # Save the model and scaler
        joblib.dump(model, 'heart_disease_model.pkl')
        joblib.dump(scaler, 'heart_scaler.pkl')
        print("Heart disease model saved to 'heart_disease_model.pkl'")
        print("Heart disease scaler saved to 'heart_scaler.pkl'")
        
    except Exception as e:
        print(f"Error training heart disease model: {e}")
        raise

if __name__ == "__main__":
    # Check if datasets exist
    if not os.path.exists('diabetes.csv'):
        print("Error: 'diabetes.csv' not found.")
        exit(1)
    
    if not os.path.exists('heart_disease_data.csv'):
        # Check if we need to rename the file
        if os.path.exists('data.csv'):
            print("Found 'data.csv', assuming this is the heart disease dataset.")
            os.rename('data.csv', 'heart_disease_data.csv')
        else:
            print("Error: Heart disease dataset not found.")
            exit(1)
    
    # Train and save both models
    train_and_save_diabetes_model()
    print()
    train_and_save_heart_disease_model()
    
    print("\nAll models have been trained and saved successfully!")
    print("These models will be automatically loaded for faster inference.") 