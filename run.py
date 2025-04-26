#!/usr/bin/env python3
"""
Medical Report Analyzer - Simplified GUI Launcher

This script launches the Medical Report Analyzer GUI application.

Usage:
    python run.py
"""

import sys
import tkinter as tk
from medical_report_analyzer import MedicalReportGUI

def main():
    """Launch the Medical Report Analyzer GUI"""
    root = tk.Tk()
    app = MedicalReportGUI(root)
    root.mainloop()
    return 0

if __name__ == "__main__":
    sys.exit(main()) 