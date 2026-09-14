import pandas as pd


def load(path: str) -> pd.DataFrame:
    """Takes the file path and returns the data loaded into a DataFrame"""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None
    except PermissionError:
        print(f"Error: Permission denied for {path}")
        return None
    except pd.errors.ParserError:
        print(f"Error: Data Parsing Error. Data might be corrupted for {path}")
        return None
    except UnicodeDecodeError:
        print(f"Error: Encoding error. File could not be decoded for {path}")
        return None
    except Exception as e:
        print(f"Unexpected error while reading file {e}")
        return None