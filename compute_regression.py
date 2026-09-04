import sys
import pandas as pd
import numpy as np
from estimate_price import estimate_price


def load(path: str) -> pd.DataFrame:
    """Takes the file path and returns the data loaded into a DataFrame"""
    try:
        df = pd.read_csv(path)
        print(f"Loading dataset of dimensions {df.shape}")
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

def calculate_theta_0(df: pd.DataFrame, learning_rate: int) -> float:
    mileage = df['km'].to_numpy()
    price = df['price'].to_numpy()

    ufunc = np.frompyfunc(estimate_price, 1, 1)
    resultat = ufunc(mileage).astype(float)
    return learning_rate * np.sum(resultat - price) / len(df.columns)

def calculate_theta_1(df: pd.DataFrame, learning_rate: int) -> float:
    mileage = df['km'].to_numpy()
    price = df['price'].to_numpy()

    ufunc = np.frompyfunc(estimate_price, 1, 1)
    resultat = ufunc(mileage).astype(float)
    return learning_rate * np.sum((resultat - price) * mileage) / len(df.columns) 

def main():
    if (len(sys.argv) != 2):
        return 0

    df = load(sys.argv[1])
    if df is None:
        return 1

    learning_rate = 0.01
    theta_0 = calculate_theta_0(df, learning_rate)
    theta_1 = calculate_theta_1(df, learning_rate)
    print(theta_0)
    print(theta_1)




if __name__ == "__main__":
    main()