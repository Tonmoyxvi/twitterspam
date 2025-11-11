import os
import subprocess

steps = [
    "src/01_preprocessing.py",
    "src/02_split_data.py",
    "src/03_train_initial_model.py",
    "src/04_evaluate_model.py",
    "src/05_tune_model.py",
    "src/06_final_test.py"
]

print("=== TWITTER SPAM DETECTION PROJECT START ===\n")

for step in steps:
    print(f"▶ Running: {step}")
    result = subprocess.run(["python3", step], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"❌ Error in {step}: {result.stderr}")
        break
    print(f"✅ Completed: {step}\n{'-'*60}\n")

print("🎯 All steps completed successfully! Check /data and /imagens folders.")
