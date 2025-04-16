import os
import pandas as pd
import numpy as np
import faiss

# File paths
train_file = "../data/processed/filter/train_split_6.parquet"
test_smiles_file = "../data/raw_filter_20/test_f_smiles.csv"
index_dir = "../data/models/KNN_Filter/KNN_f_STL"
output_file = "../data/results/predictions_KNN_STL/all_predictions_knn6.parquet"
index_file = os.path.join(index_dir, "faiss_knn_global.index")

# Create directories
os.makedirs(index_dir, exist_ok=True)
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Load data
print("Loading datasets...")
train_df = pd.read_parquet(train_file)
test_smiles = pd.read_csv(test_smiles_file)
X_train = pd.read_parquet("../data/processed/filter/X_train_f_ECFP.parquet").astype(np.float32)
X_test = pd.read_parquet("../data/processed/filter/X_test_f_ECFP.parquet").astype(np.float32)

# Check consistency
assert X_train.shape[0] == train_df.shape[0], "X_train and y_train row counts do not match"

# Extract targets and SMILES
y_train = train_df.iloc[:, 1:].replace([float('inf'), float('-inf')], pd.NA).fillna(0.0)
targets = y_train.columns
smiles = test_smiles['smiles'].tolist()

# Ensure contiguous arrays for FAISS
X_train = np.ascontiguousarray(X_train)
X_test = np.ascontiguousarray(X_test)

# Load or build FAISS index on CPU
if os.path.exists(index_file):
    print(f"Loading FAISS index from {index_file}...")
    index = faiss.read_index(index_file)
else:
    print("Building FAISS index for X_train on CPU...")
    index = faiss.IndexFlatL2(X_train.shape[1])  # L2 distance, CPU-based
    index.add(X_train)  # Add training data
    faiss.write_index(index, index_file)  # Save the index
    print(f"FAISS index saved to {index_file}")

# Precompute KNN search on CPU
k = 5
print("Precomputing nearest neighbors on CPU...")
distances, indices = index.search(X_test, k)

# Initialize predictions array
predictions_array = np.zeros((X_test.shape[0], len(targets)), dtype=np.float32)

# Check if prediction file exists and load existing progress
if os.path.exists(output_file):
    all_predictions_df = pd.read_parquet(output_file)
    if 'smiles' not in all_predictions_df.columns:
        all_predictions_df.insert(0, 'smiles', smiles)
    for i, target in enumerate(targets):
        if target in all_predictions_df.columns:
            predictions_array[:, i] = all_predictions_df[target].values
else:
    all_predictions_df = pd.DataFrame({'smiles': smiles})

# Predict for all targets with checkpoints every 50
checkpoint_interval = 50
for i, target in enumerate(targets):
    if target in all_predictions_df.columns:
        print(f"Skipping {target} ({i + 1}/{len(targets)}): Already computed")
        continue

    print(f"Processing {target} ({i + 1}/{len(targets)})...")
    y_train_target = y_train[target].values.astype(np.float32)
    predictions_array[:, i] = np.mean(y_train_target[indices], axis=1)

    # Save checkpoint every 50 targets or at the last target
    if (i + 1) % checkpoint_interval == 0 or (i + 1) == len(targets):
        all_predictions_df = pd.DataFrame(predictions_array[:, :i + 1], 
                                        columns=targets[:i + 1])
        all_predictions_df.insert(0, 'smiles', smiles)
        all_predictions_df.to_parquet(output_file)
        print(f"Checkpoint saved at {i + 1} targets to {output_file}")

print(f"All predictions saved to {output_file}")