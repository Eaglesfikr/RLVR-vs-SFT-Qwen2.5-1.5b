import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = "/data/motion-gpt/my-verl/data"

files = [
    # "gsm8k_test.parquet",
    # "sft_train.parquet",
    # "single_dsr.parquet",
    # "single_rlvr.parquet",
    #"train.parquet",
    "gsm8k_test_rlvr.parquet",
    # "sft_gsm8k_test.parquet",
]

for fname in files:
    path = f"{DATA_DIR}/{fname}"
    print("=" * 80)
    print(f"FILE: {fname}")
    print("=" * 80)

    # Read with pyarrow to see low-level schema
    pf = pq.read_table(path)
    print(f"\nSchema (pyarrow):")
    print(pf.schema)

    # Read as pandas
    df = pd.read_parquet(path)
    print(f"\nShape: {df.shape}  ({df.shape[0]} rows x {df.shape[1]} columns)")
    print(f"Columns: {list(df.columns)}")
    print(f"\nColumn dtypes:")
    for col in df.columns:
        val = df[col].iloc[0]
        print(f"  {col:30s} pandas_dtype={df[col].dtype}  python_type={type(val).__name__}")

    # Show first row with full content
    print(f"\n--- First row (all columns) ---")
    row = df.iloc[0]
    for col in df.columns:
        print(f"  [{col}]: {repr(row[col])}")

    # If there's a second row, show it too for contrast
    if len(df) > 1:
        print(f"\n--- Second row (all columns) ---")
        row = df.iloc[1]
        for col in df.columns:
            print(f"  [{col}]: {repr(row[col])}")

    print()
