import subprocess
import sys

#script to run all the other create train test sets

scripts = [
    "make_train_test_set_bert.py",
    "make_train_test_set_bow.py",
    "make_train_test_set_GloVe.py",
    "make_train_test_set_tfidf.py",
    "make_train_test_set_Word2Vec.py"
]

for script in scripts:
    print(f"\n--- Running {script} ---")
    try:
        subprocess.run([sys.executable, script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}: {e}")
