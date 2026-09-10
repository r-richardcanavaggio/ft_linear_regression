import sys
import json


def estimate_price(mileage: int, theta_0: float = 0.0, theta_1: float = 0.0) -> float:
    return theta_0 + (theta_1 * mileage)


def main():
    try:
        with open("model.json", "r") as fichier:
            parametres = json.load(fichier)

        theta_0 = parametres["theta_0"]
        theta_1 = parametres["theta_1"]
    except FileNotFoundError:
        print("Fichier model.json introuvable, lancement avec valeur par defaut (0,0)")
        theta_0 = 0.0
        theta_1 = 0.0

    mileage = input("Enter mileage: ")
    mileage = int(mileage)
    estimation = estimate_price(mileage, theta_0, theta_1)
    print(f"Estimated price for given mileage is: {estimation}")


if __name__ == "__main__":
    main()
