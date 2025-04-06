import subprocess
import sys

# List of model training scripts to run
scripts = [
    "T1_GloVe.py",
    "T1_prime BOW.py",
    "T1_prime TF_IDF.py",
    "T1_Word2Vec.py"
]

for script in scripts:
    print(f"\n--- Running {script} ---")
    try:
        subprocess.run([sys.executable, script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error occurred while running {script}: {e}")
