import sys
import pandas as pd
import numpy as np
from estimate_price import estimate_price


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


def main():
    df = load(sys.argv[1])
    if df is None:
        return 1

    learning_rate = 0.0001
    theta_0 = 0.0
    theta_1 = 0.0
    eps = 1e-12

    mileage = df['km'].to_numpy()
    mileage = (mileage - mileage.min()) / (mileage.max() - mileage.min())
    price = df['price'].to_numpy()
    m = len(df)

    while True:
        old0 = theta_0
        old1 = theta_1
        estimation = estimate_price(mileage, theta_0, theta_1)
        tmp_0 = learning_rate * np.sum(estimation - price) / m
        tmp_1 = learning_rate * (np.sum((estimation - price) * mileage) / m)
        theta_0 = theta_0 - tmp_0
        theta_1 = theta_1 - tmp_1
        print(f"Theta 0: {theta_0:.15f} | Theta 1: {theta_1:.15f}", end='\r', flush=True)
        if abs(old0 - theta_0) < eps and abs(old1 - theta_1) < eps:
            break

    theta_0 = theta_0 * (mileage.max() - mileage.min()) + mileage.min()
    theta_1 = theta_1 * (mileage.max() - mileage.min()) + mileage.min()
    print(f"\nDenormalized\nTheta 0: {theta_0} | Theta 1: {theta_1}")


if __name__ == "__main__":
    main()