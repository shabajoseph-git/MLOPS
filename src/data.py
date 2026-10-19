"""Download and standardize the UCI Heart Disease dataset."""
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "heart_disease.csv"


def download_dataset() -> pd.DataFrame:
    """Download UCI dataset 45 and save features plus binary target."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = fetch_ucirepo(id=45)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    # UCI versions may name the target 'num' or use another target column.
    if y.shape[1] != 1:
        raise ValueError(f"Expected one target column, found: {list(y.columns)}")
    target_name = y.columns[0]
    target = pd.to_numeric(y[target_name], errors="coerce")
    result = X.copy()
    # Original Cleveland-style target is 0 = no disease, 1-4 = disease.
    result["target"] = (target > 0).astype("Int64")
    result.loc[target.isna(), "target"] = pd.NA
    result = result.dropna(subset=["target"]).copy()
    result["target"] = result["target"].astype(int)
    result.to_csv(RAW_PATH, index=False)
    print(f"Saved {len(result)} rows and {len(result.columns)-1} features to {RAW_PATH}")
    print("Columns:", ", ".join(result.columns))
    print("Target counts:\n", result["target"].value_counts(dropna=False))
    return result


if __name__ == "__main__":
    download_dataset()
