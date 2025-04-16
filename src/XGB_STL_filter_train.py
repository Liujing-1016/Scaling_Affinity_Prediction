"""
XGBoost Training and Prediction Script

Trains XGBoost regression models on molecular data to predict pIC50.
Individual models x 6142 targets

Workflow:
1. Load X_train and X_test (2048-bit ECFP features).
2. Load y_train (pIC50 values for targets).
3. Train one XGBoost model per target.
4. Save and updata predictions to Parquet every 50 targets.

Inputs:
- train_target_1.parquet: SMILES and corresponding target pIC50s (float).
- test_smiles.csv: Test SMILES strings. Used for the output prediction dataframe.
- X_ecfp_train.parquet: 2048-bit ECFP features for training (float32).
- X_ecfp_test.parquet: 2048-bit ECFP features for testing (float32).

Output:
- all_predictions_xgb_1.parquet: Predicted pIC50 for all targets in `train_target_5.parquet` (float32).

Author: Liujing
Date: 2025-04-02
"""



import os
import pandas as pd
import numpy as np
import xgboost as xgb

train_file = "../data/processed/filter/train_split_6.parquet"
test_smiles = "../data/raw_filter_20/test_f_smiles.csv"
model_dir = "../data/models/XGBoost"
output_file = "../data/results/predictions_XGB_filter/predictions_xgbf_6.parquet"

os.makedirs(model_dir, exist_ok=True)

# READING DATA
print("Loading datasets...")
train_df = pd.read_parquet(train_file)
test_smiles = pd.read_csv(test_smiles)
X_train = pd.read_parquet("../data/processed/filter/X_train_f_ECFP.parquet")
X_test = pd.read_parquet("../data/processed/filter/X_test_f_ECFP.parquet")

# Check data consistency
assert X_train.shape[0] == train_df.shape[0], "X_train and y_train row counts do not match"

# Extract target variable and target columns
y_train = train_df.iloc[:, 1:]
targets = y_train.columns
smiles = test_smiles['smiles'].tolist()

# Replace inf and NA values with 0.0
y_train = y_train.replace([float('inf'), float('-inf')], pd.NA).fillna(0.0)

# Check if prediction file exists and initialize DataFrame
if os.path.exists(output_file):
    all_predictions_df = pd.read_parquet(output_file)
    if 'smiles' not in all_predictions_df.columns:
        all_predictions_df.insert(0, 'smiles', smiles)
else:
    all_predictions_df = pd.DataFrame({'smiles': smiles})

# Training and prediction
for i, target in enumerate(targets):
    print(f"\nTraining {target}...")
    y_train_target = y_train[target]

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6
    )

    model.fit(X_train, y_train_target)
    print(f"{target} training completed")

    print(f"Predicting {target}...")
    try:
        predictions = model.predict(X_test).astype(np.float32)
        all_predictions_df[target] = predictions
        # Save results every 50 targets or at the last target
        if (i + 1) % 50 == 0 or (i + 1) == len(targets):
            all_predictions_df = all_predictions_df.copy()  # Defragment DataFrame
            all_predictions_df.to_parquet(output_file)
            print(f"{target} prediction completed, {i + 1} targets processed, results saved")
        else:
            print(f"{target} prediction completed")
    except Exception as e:
        print(f"Prediction failed for {target}: {e}")
        continue

print(f"\nAll predictions have been saved to {output_file}")
