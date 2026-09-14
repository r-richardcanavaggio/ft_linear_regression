import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from estimate_price import estimate_price
from ft_load import load


def main():
    df = load(sys.argv[1])
    if df is None:
        return 1

    learning_rate = 0.0001
    theta_0 = 0.0
    theta_1 = 0.0
    eps = 1e-12

    mileage = df['km'].to_numpy()
    price = df['price'].to_numpy()

    km_min = mileage.min()
    km_max = mileage.max()
    norm_mile = (mileage - km_min) / (km_max - km_min)
    m = len(df)

    while True:
        old0 = theta_0
        old1 = theta_1
        estimation = estimate_price(norm_mile, theta_0, theta_1)
        tmp_0 = learning_rate * np.sum(estimation - price) / m
        tmp_1 = learning_rate * (np.sum((estimation - price) * norm_mile) / m)
        theta_0 = theta_0 - tmp_0
        theta_1 = theta_1 - tmp_1
        print(f"Theta 0: {theta_0:.15f} | Theta 1: {theta_1:.15f}",
              end='\r',
              flush=True)
        if abs(old0 - theta_0) < eps and abs(old1 - theta_1) < eps:
            break

    theta_1 = theta_1 / (km_max - km_min)
    theta_0 = theta_0 - (theta_1 * km_min)

    export = {
        "theta_0": theta_0,
        "theta_1": theta_1
    }

    with open("model.json", "w") as fichier:
        json.dump(export, fichier, indent=4)

    fig, ax = plt.subplots()
    ax.scatter(mileage, price, c='blue')
    ax.plot(mileage, estimate_price(mileage, theta_0, theta_1), color='red')

    plt.show()


if __name__ == "__main__":
    main()
