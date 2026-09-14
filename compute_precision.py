from ft_load import load
from estimate_price import estimate_price
import json
import sys


def main():
    df = load(sys.argv[1])
    if df is None:
        return

    try:
        with open("model.json", "r") as fichier:
            parametres = json.load(fichier)

            theta_0 = parametres["theta_0"]
            theta_1 = parametres["theta_1"]
    except FileNotFoundError:
        print("Fichier model.json introuvable,"
              "lancement avec valeur par defaut (0,0)")
        theta_0 = 0.0
        theta_1 = 0.0

    mileage = df['km'].to_numpy()
    price = df['price'].to_numpy()
    m = len(df)

    estimation = estimate_price(mileage, theta_0, theta_1)
    mean_actual = sum(price) / m
    tss = sum((price - mean_actual) ** 2)
    rss = sum((price - estimation) ** 2)
    r2 = 1 - (rss / tss)
    print(f"The regression model explains {r2 * 100:.2f}% "
          "of the variance in price.")


if __name__ == "__main__":
    main()
