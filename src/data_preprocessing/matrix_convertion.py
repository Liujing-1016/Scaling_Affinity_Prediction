import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
import argparse
import os

parser = argparse.ArgumentParser(description="Load a CSV file and perform operations.")
parser.add_argument("--file", type=str, required=True, help="Path to the input CSV file")
args = parser.parse_args()

data = pd.read_csv(args.file)
unique_compounds = data['canonical_smiles'].unique()  # unique cmpd
unique_targets = data['pref_name'].unique()          # unique target

num_samples = len(unique_compounds)  
num_targets = len(unique_targets)    

# Creating dictionary
compound_to_index = {compound: i for i, compound in enumerate(unique_compounds)}
target_to_index = {target: i for i, target in enumerate(unique_targets)}

# Initialize the matrix, 
# with the first listing SMILES and the others listing target data
matrix = np.zeros((num_samples, num_targets + 1), dtype=object) 
matrix[:, 0] = unique_compounds 

# Filling matrix
for _, row in data.iterrows():
    smiles = row['canonical_smiles']
    target = row['pref_name']
    ic50_value = row['pIC50'] 

    if pd.notna(target) and pd.notna(ic50_value):
        row_idx = compound_to_index[smiles]  
        col_idx = target_to_index[target] + 1 
        matrix[row_idx, col_idx] = float(ic50_value)

# Converting to Pandas DataFrame
columns = ['smiles'] + list(range(num_targets))
df = pd.DataFrame(matrix, columns=columns)
print("Head of the dataset:")
print(df.head())

def get_unique_filename(base_filename):
    i = 1
    filename = base_filename
    while os.path.exists(filename):
        filename = f"{base_filename.split('.')[0]}_{i}.parquet" 
        i += 1
    return filename

filename = get_unique_filename("chembl_matrix.parquet")
user_input = input("Do you want to convert into parquet file?(y/n)").strip().lower()
if user_input == 'y':
    df.to_parquet(filename, index=False)
    print(f"File has been saved as {filename}")
else: 
    print("The operation is cancelled.")



target_dict = "target_to_index.csv"

with open(target_dict, "w") as file:
    for key, value in target_to_index.items():
        file.write(f"\"{key}\",{value}\n")
print (f"The target dictionary has been exported as {target_dict} ")