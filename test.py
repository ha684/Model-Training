import pandas as pd
import sys
import os

def convert_parquet_to_csv(parquet_file, csv_file=None):
    """
    Convert a Parquet file to CSV format, preserving all columns exactly.
    
    Args:
        parquet_file: Path to the input Parquet file
        csv_file: Path to the output CSV file (optional, defaults to same name with .csv extension)
    """
    try:
        # Read the Parquet file
        df = pd.read_parquet(parquet_file)
        
        # If no output file specified, use the same name with .csv extension
        if csv_file is None:
            csv_file = os.path.splitext(parquet_file)[0] + '.csv'
        
        # Write to CSV, preserving all columns
        df.to_csv(csv_file, index=False)
        
        print(f"Successfully converted {parquet_file} to {csv_file}")
        print(f"Columns preserved: {list(df.columns)}")
        print(f"Total rows: {len(df)}")
        
    except FileNotFoundError:
        print(f"Error: File '{parquet_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <parquet_file> [csv_file]")
        print("Example: python script.py data.parquet")
        print("Example: python script.py data.parquet output.csv")
        sys.exit(1)
    
    parquet_file = sys.argv[1]
    csv_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_parquet_to_csv(parquet_file, csv_file)