import sys

def estimate_price(mileage: int, theta_0: float = 0.0, theta_1: float = 0.0) -> float:
    return theta_0 + (theta_1 * mileage)


def main():
    if len(sys.argv) != 1:
        return 1
        
    print("Enter mileage: ")
    line = sys.stdin.read()



if __name__ == "__main__":
    main()