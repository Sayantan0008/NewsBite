"""Launcher script for NewsBite dashboard with patching applied.

This script applies the necessary patches to prevent PyTorch/Streamlit
runtime errors before launching the Streamlit dashboard.
"""

import os
import sys
import subprocess

# First apply our patches
print("Applying patches to prevent PyTorch/Streamlit runtime errors...")

# Execute the entire patch file directly to ensure all patches are applied
# before any other imports that might trigger the problematic behavior
exec(open('streamlit_patch.py').read())

# Also import and apply our patching utility as a fallback
try:
    from streamlit_patch import apply_patches
    apply_patches()
except Exception as e:
    print(f"Note: Could not apply additional patches: {e}")
    # Continue anyway as the exec() should have applied the patches

# Check if Streamlit is installed
try:
    import streamlit
    print(f"Found Streamlit version: {streamlit.__version__}")
except ImportError:
    print("\nERROR: Streamlit is not installed. Please install it using:")
    print("pip install streamlit>=1.0.0,<2.0.0")
    print("\nThen run this script again.")
    sys.exit(1)

# Now launch Streamlit with the patched environment
print("\nStarting Streamlit dashboard...")

# Use subprocess to launch Streamlit with our patched environment
streamlit_cmd = [sys.executable, "-m", "streamlit", "run", "app/dashboard.py"]
process = subprocess.Popen(streamlit_cmd, cwd=os.path.dirname(__file__))

print("Streamlit process started. Press Ctrl+C to exit.")

try:
    # Keep the script running until user interrupts
    process.wait()
except KeyboardInterrupt:
    print("\nExiting...")
    process.terminate()
    process.wait()