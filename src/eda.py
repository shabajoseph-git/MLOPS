"""Create basic exploratory data analysis outputs."""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart_disease.csv"
OUT = ROOT / "artifacts" / "eda"


def run_eda() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run `python -m src.data` before EDA.")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df.describe(include="all").transpose().to_csv(OUT / "summary.csv")
    df.isna().sum().rename("missing_count").to_csv(OUT / "missing_values.csv")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="target")
    plt.title("Heart disease target class balance")
    plt.xlabel("Target (0 = absent, 1 = present in source labels)")
    plt.tight_layout()
    plt.savefig(OUT / "class_balance.png", dpi=150)
    plt.close()

    numeric = df.select_dtypes(include="number")
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Numeric feature correlation heatmap")
    plt.tight_layout()
    plt.savefig(OUT / "correlation_heatmap.png", dpi=150)
    plt.close()

    for col in [c for c in ["age", "trestbps", "chol", "thalch", "oldpeak"] if c in df]:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=col, hue="target", kde=True, element="step")
        plt.title(f"Distribution of {col} by target")
        plt.tight_layout()
        plt.savefig(OUT / f"hist_{col}.png", dpi=150)
        plt.close()
    print(f"EDA outputs saved to {OUT}")


if __name__ == "__main__":
    run_eda()
