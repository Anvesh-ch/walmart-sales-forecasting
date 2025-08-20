#!/usr/bin/env python3
"""
Streamlit Deployment Helper Script
This script helps set up and deploy the Walmart Sales Forecasting app to Streamlit Cloud.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def check_requirements():
    """Check if all required files exist."""
    required_files = [
        "app/streamlit_app.py",
        "requirements.txt",
        ".streamlit/config.toml",
        "packages.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required files present")
    return True

def test_local_app():
    """Test the Streamlit app locally."""
    print("\n🧪 Testing Streamlit app locally...")
    
    # Check if streamlit is installed
    if run_command("streamlit --version", "Checking Streamlit installation"):
        print("✅ Streamlit is installed")
    else:
        print("❌ Streamlit not found. Install with: pip install streamlit")
        return False
    
    # Test the app
    print("🚀 Starting local Streamlit app for testing...")
    print("   The app will open in your browser.")
    print("   Press Ctrl+C to stop the local server when done testing.")
    print("\n   Local URL: http://localhost:8501")
    
    try:
        subprocess.run(["streamlit", "run", "app/streamlit_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n✅ Local testing completed")
    except Exception as e:
        print(f"❌ Local testing failed: {e}")
        return False
    
    return True

def show_deployment_steps():
    """Show the deployment steps."""
    print("\n" + "="*60)
    print("🚀 STREAMLIT CLOUD DEPLOYMENT STEPS")
    print("="*60)
    
    print("\n1️⃣ Go to Streamlit Cloud:")
    print("   https://share.streamlit.io/")
    
    print("\n2️⃣ Sign in with your GitHub account")
    
    print("\n3️⃣ Click 'New app'")
    
    print("\n4️⃣ Configure your app:")
    print("   • Repository: Anvesh-ch/walmart-sales-forecasting")
    print("   • Branch: main")
    print("   • Main file path: app/streamlit_app.py")
    
    print("\n5️⃣ Click 'Deploy!'")
    
    print("\n6️⃣ Wait for build (2-5 minutes)")
    
    print("\n7️⃣ Your app will be live at the provided URL!")
    
    print("\n" + "="*60)
    print("📚 For detailed instructions, see DEPLOYMENT.md")
    print("🔗 GitHub Repository: https://github.com/Anvesh-ch/walmart-sales-forecasting")
    print("="*60)

def main():
    """Main function."""
    print("🎯 Walmart Sales Forecasting - Streamlit Deployment Helper")
    print("="*60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Cannot proceed without required files")
        sys.exit(1)
    
    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. Test the app locally")
    print("2. Show deployment steps")
    print("3. Both")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        test_local_app()
    elif choice == "2":
        show_deployment_steps()
    elif choice == "3":
        test_local_app()
        show_deployment_steps()
    elif choice == "4":
        print("👋 Goodbye!")
        return
    else:
        print("❌ Invalid choice. Please run the script again.")
        return
    
    print("\n🎉 Setup complete! Your app is ready for deployment.")

if __name__ == "__main__":
    main()
