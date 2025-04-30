"""Setup script for NewsBite project.

This script helps users install all required dependencies,
with special attention to Streamlit which is needed for the dashboard.
"""

import subprocess
import sys
import os

def check_dependency(package_name):
    """Check if a Python package is installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    print("Setting up NewsBite dependencies...\n")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_venv:
        print("WARNING: You are not running in a virtual environment.")
        print("It's recommended to use a virtual environment for this project.")
        print("You can create one with: python -m venv venv")
        print("Then activate it before running this script.\n")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    
    # Install requirements
    print("\nInstalling dependencies from requirements.txt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Specifically check for Streamlit
    if not check_dependency("streamlit"):
        print("\nStreamlit not found. Installing Streamlit specifically...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit>=1.0.0,<2.0.0"])
    
    # Verify installation
    missing_deps = []
    for dep in ["streamlit", "torch", "transformers", "pandas"]:
        if not check_dependency(dep):
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\nWARNING: The following dependencies could not be installed: {', '.join(missing_deps)}")
        print("You may need to install them manually.")
    else:
        print("\nAll dependencies installed successfully!")
        print("You can now run the dashboard with: python run_dashboard.py")

if __name__ == "__main__":
    main()