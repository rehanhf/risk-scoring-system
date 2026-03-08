import numpy as np
import pandas as pd


def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    # 1. define static bin boundaries berdasarkan bins, pada distribusi (Train) yang diharapkan
    breakpoints = np.histogram_bin_edges(expected, bins=bins)

    # 2. kalkulasi proportions di setiap bin
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # 3. mencegah pembagian nol mathematically
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    # 4. Compute PSI
    psi_values = (actual_percents - expected_percents) * np.log(
        actual_percents / expected_percents
    )
    return float(np.sum(psi_values))


def simulate_drift(train_path: str, test_path: str):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    feature = "LIMIT_BAL"

    # skenario a: normal out-of-time data (No Drift)
    psi_normal = calculate_psi(train[feature], test[feature])

    # skenario b: macroeconomic shock (Drift)
    # semulasi resesi dimana bank memotong credit limits drastically
    test_drifted = test.copy()
    test_drifted[feature] = test_drifted[feature] * 0.3

    psi_drifted = calculate_psi(train[feature], test_drifted[feature])

    print(f"Feature: {feature}")
    print(f"PSI (Normal OOT): {psi_normal:.4f}")
    print(f"PSI (Recession Simulation): {psi_drifted:.4f}")

    if psi_drifted > 0.2:
        print(
            "CRITICAL ALERT: PSI > 0.2. Data distribution has severely drifted. Retraining required."
        )


if __name__ == "__main__":
    simulate_drift("data/processed/train.csv", "data/processed/test.csv")
