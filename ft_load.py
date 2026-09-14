import pandas as pd


def load(path: str):
    """Load dataset from CSV and return mileage and price arrays."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        return None
    except PermissionError:
        print(f"Error: Permission denied for '{path}'.")
        return None
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        print(f"Error: Could not parse or decode '{path}': {e}")
        return None
    except Exception as e:
        print(f"Unexpected error reading '{path}': {e}")
        return None

    try:
        data_array = df.to_numpy().T

        mileage, price = data_array

        return mileage, price

    except ValueError:
        print("Error: Data shape mismatch. "
              f"Expected 2 columns, found {df.shape[1]}.")
        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None
