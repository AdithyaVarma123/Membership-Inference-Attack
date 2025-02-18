import os
import tarfile
import numpy as np
import tensorflow as tf
from scipy.stats import entropy

# ------------------------------
# Step 1: Extract Dataset
# ------------------------------
dataset_path = "dataset_texas.tgz"

if not os.path.exists("dataset_texas"):
    with tarfile.open(dataset_path, "r:gz") as tar:
        tar.extractall()

# Load features and labels (Ensure correct delimiter)
features = np.loadtxt("dataset_texas/texas/100/feats", delimiter=",")
labels = np.loadtxt("dataset_texas/texas/100/labels", delimiter=",")

# Consider only the first 200 records
features = features[:200]
labels = labels[:200]

print("[INFO] Dataset loaded and preprocessed.")

# ------------------------------
# Step 2: Load the Model using TensorFlow SavedModel
# ------------------------------
model_path = "trained_model"

try:
    model = tf.saved_model.load(model_path)
    print("[INFO] Model loaded as a TensorFlow SavedModel.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit()

# Extract the inference function
infer = model.signatures["serving_default"]

# Get the correct output tensor dynamically
output_tensor_name = list(infer.structured_outputs.keys())[0]  # Get first output key dynamically
print(f"[INFO] Detected output tensor: {output_tensor_name}")

# ------------------------------
# Step 3: Make Predictions & Compute Confidence Scores
# ------------------------------
features_tensor = tf.constant(features, dtype=tf.float32)

# Run inference using the correct output tensor name
predictions = infer(features_tensor)[output_tensor_name].numpy()

# Compute confidence (max softmax probability) for each sample
confidences = np.max(predictions, axis=1)

# Compute entropy of softmax output
entropies = np.apply_along_axis(lambda x: entropy(x), 1, predictions)

# Compute ranking score (higher confidence, lower entropy preferred)
ranking_score = confidences - entropies  

# Rank records and get top 100 most likely members
member_indices = np.argsort(ranking_score)[-100:]

print("[INFO] Membership inference attack completed.")

# ------------------------------
# Step 4: Save Results
# ------------------------------

# Save identified members to members.txt
with open("members.txt", "w") as f:
    f.write(str(list(member_indices)))

# Generate a brief report
report_content = f"""
Membership Inference Attack Report
----------------------------------
- Dataset: Texas-100 (first 200 records considered)
- Model: 4-layer dense neural network (Loaded as TensorFlow SavedModel)
- Attack Method: Confidence Score + Entropy Ranking
- Selected Top 100 Most Likely Members
- Output File: members.txt

Methodology:
We used a ranking approach where we computed the confidence score (maximum softmax probability)
and subtracted the entropy of the softmax distribution. Higher confidence and lower entropy indicate
that a sample is more likely to have been in the training set.

Results:
The IDs of the inferred training samples are saved in 'members.txt'.
"""

# Save report
with open("report.txt", "w") as f:
    f.write(report_content)

print("[INFO] Results saved in members.txt and report.txt.")

# ------------------------------
# Final Output
# ------------------------------
print("\nTop 10 Most Likely Members (IDs):", member_indices[-10:])
print("\n[PROCESS COMPLETED]")

