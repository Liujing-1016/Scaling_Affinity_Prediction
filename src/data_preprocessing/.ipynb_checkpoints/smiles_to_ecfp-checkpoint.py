import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_ecfp(smiles_list, radius=2, n_bits=2048):
    """
    Convert a list of SMILES to ECFP fingerprints.

    Parameters:
    - smiles_list (list): List of SMILES strings.
    - radius (int): Radius for Morgan fingerprint (default: 2).
    - n_bits (int): Number of bits in the fingerprint (default: 2048).

    Returns:
    - np.ndarray: A 2D NumPy array of ECFP fingerprints.
    """
    ecfp_features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            # Generate ECFP fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            ecfp_features.append(np.array(fp, dtype=np.float32))  # Convert to float32
        else:
            # If SMILES is invalid, append a zero vector
            ecfp_features.append(np.zeros(n_bits, dtype=np.float32))  # Ensure consistent data type
    
    return np.vstack(ecfp_features)  # Stack into a 2D NumPy array

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert SMILES to ECFP fingerprints and save as Parquet file.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input Parquet file.")
    parser.add_argument("--output", type=str, default="X_test.parquet", help="Path to save the output Parquet file (default: X_test.parquet).")
    parser.add_argument("--radius", type=int, default=2, help="Radius for Morgan fingerprint (default: 2).")
    parser.add_argument("--n_bits", type=int, default=2048, help="Number of bits in the fingerprint (default: 2048).")
    args = parser.parse_args()

    # Load the input Parquet file
    df = pd.read_parquet(args.input)

    # Convert SMILES to ECFP fingerprints
    smiles_list = df.iloc[:, 0].tolist()  # Assume SMILES are in the first column
    X = smiles_to_ecfp(smiles_list, radius=args.radius, n_bits=args.n_bits)

    # Save the ECFP fingerprints as a Parquet file
    pd.DataFrame(X).to_parquet(args.output, index=False)
    print(f"ECFP fingerprints saved to {args.output}")

if __name__ == "__main__":
    main()