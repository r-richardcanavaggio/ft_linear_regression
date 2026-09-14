from estimate_price import estimate_price
import json
import pandas as pd
import numpy as np


def run_precision(
    values: dict, mileage: np.ndarray, price: np.ndarray
) -> float:
    """Compute the R2 determination coefficient as a percentage."""
    theta_0 = values['theta_0']
    theta_1 = values['theta_1']

    m = len(mileage)

    estimation = estimate_price(mileage, theta_0, theta_1)
    mean_actual = sum(price) / m
    tss = sum((price - mean_actual) ** 2)
    rss = sum((price - estimation) ** 2)
    r2 = 1 - (rss / tss)

    return r2 * 100


def main():
    """Load data and model parameters, then print precision."""
    mileage, price = pd.read_csv(
        'linear_regression_data.csv'
    ).to_numpy().T

    try:
        with open("model.json", "r") as fichier:
            values = json.load(fichier)
    except FileNotFoundError:
        print("Error: Missing 'model.json'")
        return

    r2 = run_precision(values, mileage, price)
    print(f"The regression model explains {r2:.2f}% "
          "of the variance in price.")


if __name__ == "__main__":
    main()
