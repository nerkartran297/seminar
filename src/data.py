from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import urllib.request


UCI_RED_WINE_URLS: Tuple[str, ...] = (
    # Primary (legacy) UCI archive URL
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    # Backup mirrors (in case UCI path changes)
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv?raw=1",
)


def _ensure_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def _download_with_fallback(urls: Iterable[str], dest_path: Path) -> None:
    last_error: Exception | None = None
    _ensure_dir(dest_path.parent)
    for url in urls:
        try:
            urllib.request.urlretrieve(url, dest_path.as_posix())
            return
        except Exception as exc:  # noqa: BLE001 - surfacing last error below
            last_error = exc
            continue
    raise RuntimeError(
        f"Failed to download dataset to {dest_path} from provided URLs. Last error: {last_error}"
    )


def load_winequality_red(data_dir: str = "data", filename: str = "winequality-red.csv") -> pd.DataFrame:
    """Load the red wine quality dataset as a pandas DataFrame.

    If the CSV is not found in `data_dir`, attempts to download it from UCI.
    The CSV uses semicolon separators.
    """
    data_dir_path = Path(data_dir)
    csv_path = data_dir_path / filename

    if not csv_path.exists():
        _download_with_fallback(UCI_RED_WINE_URLS, csv_path)

    # UCI file uses semicolon separators
    df = pd.read_csv(csv_path, sep=";")
    return df


def make_quality_labels(quality: pd.Series, scheme: str = "3class") -> pd.Series:
    """Map numeric quality scores to categorical labels.

    scheme="3class": poor (<=4), average (5-6), good (>=7)
    Returns a pandas Series of dtype category with ordered classes.
    """
    if scheme != "3class":
        raise ValueError("Only '3class' scheme is supported currently.")

    labels = (
        quality
        .apply(lambda q: "poor" if q <= 4 else ("average" if q <= 6 else "good"))
        .astype("category")
    )
    labels = labels.cat.set_categories(["poor", "average", "good"], ordered=True)
    return labels


