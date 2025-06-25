"""
KNN Training and Prediction Script (Optimized with Index Saving)

Performs KNN regression on molecular data to predict pIC50.
Individual models for targets in a subset (Single Task Learning).

Workflow:
1. Load X_train and X_test (2048-bit ECFP features).
2. Load y_train (pIC50 values for targets in subset).
3. Build or load a single FAISS KNN index for all targets.
4. Use the index to predict for each target.
5. Save and update predictions to Parquet every 50 targets.

Inputs:
- train_split_1.parquet: SMILES and subset of target pIC50s (float).
- test_f_smiles.csv: Test SMILES strings.
- X_train_f_ECFP.parquet: 2048-bit ECFP features for training (float32).
- X_test_f_ECFP.parquet: 2048-bit ECFP features for testing (float32).

Output:
- all_predictions_knn1.parquet: Predicted pIC50 for subset targets (float32).

Author: Liujing
Date: 2025-03-24
"""

import os
import pandas as pd
import numpy as np
import faiss

# File paths
train_file = "../data/processed/filter/train_split_1.parquet"  # 子集 1，可改为 2-6
test_smiles_file = "../data/raw_filter_20/test_f_smiles.csv"
index_dir = "../data/models/KNN_Filter/KNN_f_STL"
output_file = "../data/results/predictions_KNN_STL/all_predictions_knn1.parquet"
index_file = os.path.join(index_dir, "faiss_knn_global.index")  # 索引保存路径

# Create directories
os.makedirs(index_dir, exist_ok=True)
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# READING DATA
print("Loading datasets...")
train_df = pd.read_parquet(train_file)
test_smiles = pd.read_csv(test_smiles_file)
X_train = pd.read_parquet("../data/processed/filter/X_train_f_ECFP.parquet")
X_test = pd.read_parquet("../data/processed/filter/X_test_f_ECFP.parquet")

# Check data consistency
assert X_train.shape[0] == train_df.shape[0], "X_train and y_train row counts do not match"

# Extract target variables and columns
y_train = train_df.iloc[:, 1:]  # All columns except SMILES
targets = y_train.columns
smiles = test_smiles['smiles'].tolist()

# Replace inf and NA values with 0.0
y_train = y_train.replace([float('inf'), float('-inf')], pd.NA).fillna(0.0)

# Ensure X_train and X_test are float32 for FAISS
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# Load or build FAISS index
if os.path.exists(index_file):
    print(f"Loading existing FAISS index from {index_file}...")
    index = faiss.read_index(index_file)
    print(f"FAISS index loaded with {index.ntotal} samples")
else:
    print("Building FAISS index for X_train...")
    index = faiss.IndexFlatL2(X_train.shape[1])  # L2 distance, 2048 dimensions
    index.add(X_train)  # Add training data once
    faiss.write_index(index, index_file)  # Save the index
    print(f"FAISS index built and saved to {index_file} with {index.ntotal} samples")

# Check if prediction file exists and initialize DataFrame
if os.path.exists(output_file):
    all_predictions_df = pd.read_parquet(output_file)
    if 'smiles' not in all_predictions_df.columns:
        all_predictions_df.insert(0, 'smiles', smiles)
else:
    all_predictions_df = pd.DataFrame({'smiles': smiles})

# KNN Prediction with FAISS
k = 5  # Number of nearest neighbors (adjustable)
for i, target in enumerate(targets):
    print(f"\nProcessing {target} (Target {i + 1}/{len(targets)})...")
    y_train_target = y_train[target].values.astype(np.float32)

    # Perform KNN search using the pre-built/loaded index
    distances, indices = index.search(X_test, k)

    # Compute predictions as mean of k-nearest neighbors' target values
    predictions = np.mean(y_train_target[indices], axis=1).astype(np.float32)

    # Add predictions to DataFrame
    all_predictions_df[target] = predictions

    # Save results every 50 targets or at the last target
    if (i + 1) % 50 == 0 or (i + 1) == len(targets):
        all_predictions_df = all_predictions_df.copy()  # Defragment DataFrame
        all_predictions_df.to_parquet(output_file)
        print(f"{target} processed, {i + 1} targets completed, results saved to {output_file}")
    else:
        print(f"{target} processed")

print(f"\nAll predictions have been saved to {output_file}")