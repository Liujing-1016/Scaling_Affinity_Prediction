# ChEMBL Data Preparation

This is a collection of Python scripts for extracting, converting, and splitting chemical data.

## WorkFlow

## 1. Extract Target-Compound Pairs from ChEMBL
- **Script**: `data_extraction.py`
- **Purpose**: Extract all available target-compound pair entries from ChEMBL.
```bash
python data_extraction.py --dp_path='/chembl_34/chembl_34_sqlite/chembl_34.db'
```
- **Output**: Generates `pIC50_chembl.csv` in the current directory.

## 2. Convert to Matrix and Dictionary
- **Script**: `matrix_conversion.py`
- **Purpose**: Convert `pIC50_chembl.csv` into a matrix format and create a target-index dictionary.
```bash
python matrix_conversion.py --file='/MolE_Evaluation_Project/data/raw/pIC50_chembl.csv'
```
- This script converts the specified file into a matrix format. The required file should be specified using `--file=`.

- The default generated output file is `chembl_matrix.csv`, a matrix of ~770k compounds x ~6k targets, where each row represents a compound SMILES and each column represents a specific target. The filename will not be overwritten. A corresponding `target dictionary` is also created.

## 3. Manually Remove Invalid Targets

- Remove columns with invalid targets from the `chembl_matrix.csv`, such as those marked as "unchecked". Update `target dictionary`. 

## 4. Split the Matrix into Train, Validation, and Test Sets
- **Script**: `dataset_split.py`
- 
- This script splits the input dataset into two subsets and is used to split the preprocessed matrix into train, validation, and test sets.

```bash
python dataset_split.py --input_file='/home/jovyan/proj-liujing/random/random_smiles.csv' --test_file='t1.csv' --train_file='t2.csv'
python dataset_split.py --input_file='/home/jovyan/proj-liujing/random/random_smiles.csv'
```
- Optional parameters:
`--test_size=` (default: 0.2)
`--random_state=` (default: 42)

- If output filenames (`--test_file` and `--train_file`) are not specified, the existing filenames will be overwritten.


## 5. Generate ECFP for X_train, X_test
- **Script**: `ECFP_convertion.py`

- **Purpose**: Extract a list of SMILES and convert it to ECFP fingerprints from X_train and X_test. 

```bash
python smiles_to_ecfp.py --input /path/to/input.parquet --output /path/to/output.parquet --radius 2 --n_bits 2048
```

- input: Specifies the path to enter the Parquet file.
- output: Specifies the path to the output Parquet file (optional, default is X_test.parquet).
- radius: Specifies the radius of Morgan's fingerprint (optional. The default value is 2).
- n_bits: Specifies the number of bits for a fingerprint. (Optional. The default value is 2048.)

## 6. Split the datasets for XGBoost single-task models training

This script splits a large `.parquet` file into training and testing sets in a memory-efficient, chunk-wise manner. It is particularly useful for large-scale datasets that cannot be loaded entirely into memory.

- Reads the input dataset row group by row group using `pyarrow`
- Concatenates data into chunks of specified size
- Splits each chunk into train and test subsets using `train_test_split`
- Saves the output subsets incrementally to `.parquet` files



## 7. Filtered Subset Creation
- **Data Filtering Criteria**
To ensure that each molecular target (identified by the pref_name column) has sufficient data for robust model training and evaluation, we applied the following two-step filtering procedure:

- **(1)Minimum Target Representation**:
We retained only targets (pref_name) that occur at least 20 times in the dataset, i.e., targets with ≥ 20 molecule entries.

- **(2)Minimum Active Ligand Count**:
Among those, we further required that each target must have at least 20 molecules with pIC50 > 5, indicating a biologically meaningful level of activity.

Only targets that satisfied both conditions were included in the final dataset. This filtering step ensures that all retained targets are both sufficiently represented and have a minimum number of active compounds, improving training stability and fair model comparison.

## 8. Data Visualizaton by Target Types