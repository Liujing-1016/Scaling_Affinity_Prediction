import sqlite3
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description="Export SQL query results to CSV")
parser.add_argument("--db_path", default='/home/jovyan/proj-liujing/chembl_34/chembl_34_sqlite/chembl_34.db', help="Path to the database file.")

args = parser.parse_args()
db_path=args.db_path

def get_unique_filename(base_filename):
    i = 1
    filename = base_filename
    while os.path.exists(filename):
        filename = f"{base_filename.split('.')[0]}_{i}.csv" 
        i += 1
    return filename

def export_query(query, db_path):
    """
    Execute SQL queries and export the results from ChEMBL 
    as CSV files.

    Parameters:
    - query (str): SQL query statement
    - dataset (str): indicates the exported CSV file name (including the path).
    -db_path (str): database file path, default is 'chembl.db'
    """

    try:
        conn = sqlite3.connect(db_path)
        data = pd.read_sql_query(query, conn)
        print(data.head())
        #os.chdir('/home/jovyan/proj-liujing/Data_preparation')
        
        user_input = input("Do you want to convert into csv file?(y/n)").strip().lower()
        if user_input == 'y':
            filename = get_unique_filename("pIC50_ChEMBL.csv")
            data.to_csv(filename, index=False)
            print(f"Data has been exported from ChEMBL and saved as {filename}")       
            print("Unique targets:", data['pref_name'].nunique())
            print("Unique compounds:", data['canonical_smiles'].nunique())
        else: 
            print("The operation is cancelled.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

query1 = '''
SELECT cs.molregno, cs.canonical_smiles, t.pref_name, a.standard_value AS ic50, a.pchembl_value AS pIC50, a.standard_units AS units
FROM compound_structures cs
JOIN activities a ON cs.molregno = a.molregno
JOIN assays ass ON a.assay_id = ass.assay_id
JOIN target_dictionary t ON ass.tid = t.tid
WHERE a.standard_type = 'IC50'
AND a.pchembl_value IS NOT NULL;
'''
export_query(query1,db_path)