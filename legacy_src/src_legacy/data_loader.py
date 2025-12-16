import pandas as pd
import os

def load_csv(path):
    """
    Load a CSV file from the given path and return a pandas DataFrame.
    Raises an error if the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)
