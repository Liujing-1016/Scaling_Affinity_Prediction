import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import pyarrow as pa
import pyarrow.parquet as pq
import os

parser = argparse.ArgumentParser(description="Split dataset into train and test files.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input parquet file.")
parser.add_argument("--train_file", type=str, default='train_df.parquet', help="Path to save the train parquet file.")
parser.add_argument("--test_file", type=str, default='test_df.parquet', help="Path to save the test parquet file.")
parser.add_argument("--chunksize", type=int, default=100000, help="Number of rows per chunk. Default is 100000.")
parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split. Default is 0.2.")
parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility. Default is 42.")

args = parser.parse_args()

if not os.path.exists(args.input_file):
    raise FileNotFoundError(f"Input file '{args.input_file}' does not exist.")

# Open the Parquet file
parquet_file = pq.ParquetFile(args.input_file)
num_rows = parquet_file.metadata.num_rows
print(f"Total rows in file: {num_rows}")

# Initialize writers and buffer
train_writer = None
test_writer = None
buffer = pd.DataFrame()

for i in range(parquet_file.num_row_groups):
    table = parquet_file.read_row_group(i)
    chunk = table.to_pandas()
    buffer = pd.concat([buffer, chunk], ignore_index=True)

    # Process in chunks of specified size
    while len(buffer) >= args.chunksize:
        chunk_to_process = buffer.iloc[:args.chunksize]
        buffer = buffer.iloc[args.chunksize:]

        train_chunk, test_chunk = train_test_split(chunk_to_process, test_size=args.test_size, random_state=args.random_state)
        
        # Converting DataFrame to PyArrow and remove index
        train_table = pa.Table.from_pandas(train_chunk, preserve_index=False)
        test_table = pa.Table.from_pandas(test_chunk, preserve_index=False)

        if train_writer is None:
            train_writer = pq.ParquetWriter(args.train_file, train_table.schema)
            test_writer = pq.ParquetWriter(args.test_file, test_table.schema)

        train_writer.write_table(train_table)
        test_writer.write_table(test_table)

# Process any remaining rows
if not buffer.empty:
    train_chunk, test_chunk = train_test_split(buffer, test_size=args.test_size, random_state=args.random_state)
    
    # Converting DataFrame to PyArrow and remove index
    train_table = pa.Table.from_pandas(train_chunk, preserve_index=False)
    test_table = pa.Table.from_pandas(test_chunk, preserve_index=False)

    if train_writer is None:
        train_writer = pq.ParquetWriter(args.train_file, train_table.schema)
        test_writer = pq.ParquetWriter(args.test_file, test_table.schema)

    train_writer.write_table(train_table)
    test_writer.write_table(test_table)

# Close writers
train_writer.close()
test_writer.close()

print(f"Data has been split and saved to {args.train_file} and {args.test_file}!")