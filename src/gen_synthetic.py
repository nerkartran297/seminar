from __future__ import annotations

import argparse
import csv
from math import exp
from pathlib import Path
from typing import List
import random as pyrandom


HEADERS: List[str] = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
]


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _round(value: float, ndigits: int) -> float:
    return float(f"{value:.{ndigits}f}")


def sample_row(rng: pyrandom.Random) -> List[float]:
    # Base features sampled from plausible ranges (red wine)
    fixed_acidity = _clip(rng.gauss(7.9, 1.5), 4.6, 15.9)
    volatile_acidity = _clip(rng.gauss(0.52, 0.18), 0.12, 1.58)
    citric_acid = _clip(rng.gauss(0.27, 0.16), 0.0, 1.00)
    residual_sugar = _clip(rng.gauss(2.5, 1.5), 0.9, 15.5)
    chlorides = _clip(rng.gauss(0.087, 0.04), 0.012, 0.611)
    free_so2 = int(_clip(rng.gauss(16.0, 10.0), 1, 72))
    total_so2 = int(_clip(free_so2 + rng.uniform(20, 160), 6, 289))

    # Derived correlations
    # density ~ base + sugar contribution - alcohol contribution + noise
    # Start from typical mean 0.9967
    base_density = 0.9967
    density = (
        base_density
        + 0.0009 * (residual_sugar - 2.5)
        - 0.0004 * (10.5 - 10.4)  # small centering constant
    )

    # sulphates modestly independent but beneficial for quality
    sulphates = _clip(rng.gauss(0.66, 0.17), 0.33, 2.0)

    # alcohol ~ slightly influenced by residual sugar (fermentation) + noise
    alcohol = _clip(rng.gauss(10.4 + 0.05 * (residual_sugar - 2.5), 1.0), 8.4, 14.9)

    # refine density with alcohol inverse effect and small noise
    density = _clip(density - 0.0006 * (alcohol - 10.4) + rng.gauss(0.0, 0.0006), 0.9900, 1.0040)

    # pH inversely related to fixed acidity with noise
    pH = _clip(3.4 - 0.06 * (fixed_acidity - 7.0) + rng.gauss(0.0, 0.05), 2.74, 4.01)

    # Latent quality score as a weighted sum with noise
    latent = (
        + 0.7 * (alcohol - 10.4)
        - 0.8 * (volatile_acidity - 0.52)
        + 0.4 * (sulphates - 0.66)
        + 0.2 * (citric_acid - 0.27)
        - 0.15 * (chlorides - 0.087)
        - 0.05 * (density - 0.9967) * 1000.0
        + 0.05 * (free_so2 - 16.0) / 10.0
        - 0.03 * (total_so2 - 46.0) / 50.0
        + rng.gauss(0.0, 0.6)
    )

    # Map latent to quality 3..8 approximately
    # Use logistic-like squashing for stability
    scaled = 0.5 + 2.5 * (latent)
    raw_quality = 5.5 + scaled
    quality = int(_clip(round(raw_quality), 3, 8))

    # Format rounding similar to dataset
    row = [
        _round(fixed_acidity, 1),
        _round(volatile_acidity, 3),
        _round(citric_acid, 3),
        _round(residual_sugar, 1),
        _round(chlorides, 3),
        int(free_so2),
        int(total_so2),
        _round(density, 4),
        _round(pH, 2),
        _round(sulphates, 2),
        _round(alcohol, 1),
        int(quality),
    ]
    return row


def generate(rows: int, seed: int) -> List[List[float]]:
    rng = pyrandom.Random(seed)
    return [sample_row(rng) for _ in range(rows)]


def write_csv(rows: List[List[float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(HEADERS)
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic winequality-red.csv")
    parser.add_argument("--rows", type=int, default=700, help="Number of rows to generate")
    parser.add_argument("--out", type=str, default="data/winequality-red.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = generate(rows=args.rows, seed=args.seed)
    write_csv(rows, Path(args.out))
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()


