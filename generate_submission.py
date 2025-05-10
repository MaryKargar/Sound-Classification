# generate_submission.py
import os
import pandas as pd

# Automatically find the latest experiment directory
runs_path = 'results'
experiment_root = max(
    [os.path.join(runs_path, d) for d in os.listdir(runs_path)],
    key=os.path.getmtime
)

print(f"Using latest experiment directory: {experiment_root}")

# Set path to single combined test_probs_terminal.csv
probs_path = os.path.join(experiment_root, 'test_probs_terminal.csv')

# Check if file exists
if not os.path.exists(probs_path):
    raise FileNotFoundError(f"Missing file: {probs_path}")

# Read prediction probabilities
df = pd.read_csv(probs_path, index_col=0)

# Ensure the first column has no name (=> CSV header will be 'null')
df.index.name = None

# Save submission in correct format
df.to_csv('submission_test.csv', index=True)
print(" submission_test.csv created successfully!")
