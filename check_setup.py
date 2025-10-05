"""
Quick setup checker to diagnose issues
Run this first: python check_setup.py
"""
import os
import sys

print("="*70)
print("STOCK PRICE PREDICTION - SETUP CHECKER")
print("="*70)

# Check current directory
current_dir = os.getcwd()
print(f"\nCurrent Directory: {current_dir}")

# Check directory structure
print("\n" + "-"*70)
print("CHECKING PROJECT STRUCTURE:")
print("-"*70)

required_structure = {
    'data': ['aapl_stock.csv'],
    'src': [
        'preprocess.py',
        'train.py',
        'preprocess_lstm.py',
        'train_lstm.py',
        'evaluate_lstm.py',
        'compare_models.py'
    ],
    'models': []  # Will be created
}

all_good = True

for folder, files in required_structure.items():
    folder_path = os.path.join(current_dir, folder)
    
    if os.path.exists(folder_path):
        print(f"\n✓ {folder}/ folder exists")
        
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"  ✓ {file} ({size} bytes)")
            else:
                print(f"  ❌ {file} - MISSING!")
                all_good = False
    else:
        print(f"\n❌ {folder}/ folder - MISSING!")
        all_good = False

# Check if models exist
models_dir = os.path.join(current_dir, 'models')
if os.path.exists(models_dir):
    model_files = os.listdir(models_dir)
    if model_files:
        print(f"\n✓ models/ folder exists with files:")
        for f in model_files:
            print(f"  - {f}")
    else:
        print(f"\n⚠ models/ folder exists but is empty (models not trained yet)")
else:
    print(f"\n⚠ models/ folder doesn't exist (will be created during training)")

# Check Python packages
print("\n" + "-"*70)
print("CHECKING PYTHON PACKAGES:")
print("-"*70)

required_packages = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'sklearn': 'scikit-learn',
    'matplotlib': 'matplotlib',
    'tensorflow': 'tensorflow',
    'joblib': 'joblib (usually comes with scikit-learn)'
}

for module, package in required_packages.items():
    try:
        __import__(module)
        print(f"✓ {package}")
    except ImportError:
        print(f"❌ {package} - NOT INSTALLED!")
        print(f"   Install with: pip install {package}")
        all_good = False

# Test imports from src
print("\n" + "-"*70)
print("CHECKING CUSTOM MODULES:")
print("-"*70)

sys.path.insert(0, os.path.join(current_dir, 'src'))

modules_to_test = [
    ('preprocess', 'preprocess.py'),
    ('preprocess_lstm', 'preprocess_lstm.py'),
]

for module_name, file_name in modules_to_test:
    try:
        module = __import__(module_name)
        print(f"✓ {file_name} - imports successfully")
    except ImportError as e:
        print(f"❌ {file_name} - import failed!")
        print(f"   Error: {str(e)}")
        all_good = False
    except Exception as e:
        print(f"⚠ {file_name} - has issues: {str(e)}")

# Final verdict
print("\n" + "="*70)
if all_good:
    print("🎉 ALL CHECKS PASSED!")
    print("\nYou're ready to go! Next steps:")
    print("  1. python src/train.py              # Train Linear Regression")
    print("  2. python src/train_lstm.py         # Train LSTM")
    print("  3. python src/compare_models.py     # Compare both models")
else:
    print("⚠️ SOME ISSUES FOUND!")
    print("\nPlease fix the issues above and run this check again.")
print("="*70)