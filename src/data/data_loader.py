import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load the housing dataset from a CSV file and strip column names.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def show_overview(df: pd.DataFrame):
    """
    Print dataset shape, memory usage, and a sample.
    """
    print(f"Dataset loaded: {df.shape[0]} rows and {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print("\nSample of the data:")
    print(df.head())

def initial_assessment(df: pd.DataFrame):
    """
    Print initial data assessment: dimensions, types, missing, duplicates, info.
    """
    print("Initial Data Assessment")
    print("=" * 40)
    print(f"Dataset dimensions: {df.shape}")
    print(f"Data types:\n{df.dtypes.value_counts()}")
    print(f"\nData Quality Check:")
    print(f"Total cells: {df.shape[0] * df.shape[1]:,}")
    print(f"Missing values: {df.isnull().sum().sum():,} ({df.isnull().sum().sum()/(df.shape[0] * df.shape[1])*100:.1f}%)")
    print(f"Duplicate rows: {df.duplicated().sum():,}")
    print("\nQuick data overview:")
    print(df.info())