import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from estimate_price import estimate_price
from ft_load import load


def run_regression(mileage: np.ndarray, price: np.ndarray) -> dict:
    """Train linear regression model on mileage and price data."""
    learning_rate = 0.001
    theta_0 = 0.0
    theta_1 = 0.0
    eps = 1e-10

    km_min = mileage.min()
    km_max = mileage.max()
    norm_mile = (mileage - km_min) / (km_max - km_min)
    m = len(mileage)

    while True:
        old0 = theta_0
        old1 = theta_1
        estimation = estimate_price(norm_mile, theta_0, theta_1)
        tmp_0 = learning_rate * np.sum(estimation - price) / m
        tmp_1 = learning_rate * (np.sum((estimation - price) * norm_mile) / m)
        theta_0 = theta_0 - tmp_0
        theta_1 = theta_1 - tmp_1
        print(
            f"Theta 0: {theta_0:.15f} | Theta 1: {theta_1:.15f}",
            end='\r',
            flush=True,
        )
        if abs(old0 - theta_0) < eps and abs(old1 - theta_1) < eps:
            break

    theta_1 = theta_1 / (km_max - km_min)
    theta_0 = theta_0 - (theta_1 * km_min)

    export = {
        "theta_0": theta_0,
        "theta_1": theta_1
    }

    return export


def load_computed_results(file_path: Path) -> dict | None:
    """Load previously computed model parameters from a JSON file."""
    print(f"File found: '{file_path}. Loading existing results.")
    try:
        with open(file_path, 'r') as fichier:
            return json.load(fichier)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading file: {e}, Re-running computation")
        return None


def plot(mileage: np.ndarray, price: np.ndarray, values: dict) -> None:
    """Plot data points alongside the fitted regression line."""
    print("Plotting...")

    theta_0 = values['theta_0']
    theta_1 = values['theta_1']
    fig, ax = plt.subplots()
    ax.scatter(mileage, price, c='blue')
    ax.plot(mileage, estimate_price(mileage, theta_0, theta_1), color='red')

    plt.show()


def main():
    """Train or load the linear regression model and optionally plot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plot", action="store_true")
    args = parser.parse_args()

    file_path = Path('model.json')
    computed_data = None

    path = 'linear_regression_data.csv'
    result = load(path)

    if result:
        mileage, price = result
        print("Data loaded successfully.")
    else:
        print("Program stopping due to data loading failure.")
        return

    if file_path.exists():
        computed_data = load_computed_results(file_path)
    else:
        computed_data = run_regression(mileage, price)

        try:
            with open(file_path, 'w') as fichier:
                json.dump(computed_data, fichier, indent=4)
            print("Successfully saved new results to 'model.json'")
        except IOError as e:
            print(f"Error: Could not write to file {file_path} ({e})")
            return

    if args.plot:
        plot(mileage, price, computed_data)


if __name__ == "__main__":
    main()
