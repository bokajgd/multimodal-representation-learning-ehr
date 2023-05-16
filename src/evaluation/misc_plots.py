"""Script for misc. plots and tables."""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DF_PATH = (
    PROJECT_ROOT / "outputs" / "model_outputs" / "model_outputs_05_13_2023_00_41_54"
)
eval_df = pd.read_csv(EVAL_DF_PATH / "evaluation_dataset.csv")


def plot_days_to_outcome_distribution(eval_df: pd.DataFrame = eval_df) -> None:
    """Plot the distribution of days between prediction and outcome.

    Args:
        eval_df (pd.DataFrame, optional): The evaluation dataset. Defaults to eval_df.

    Returns:
        None
    """

    eval_df = eval_df.dropna(subset=["eval_dod"])

    eval_df["pred_timestamps"] = pd.to_datetime(eval_df["pred_timestamps"])
    eval_df["eval_dod"] = pd.to_datetime(eval_df["eval_dod"])

    # calc the number of days between prediction and outcome
    eval_df["days_to_outcome"] = (
        eval_df["eval_dod"] - eval_df["pred_timestamps"]
    ).dt.days

    bins = eval_df["days_to_outcome"].nunique()

    sns.histplot(data=eval_df, x="days_to_outcome", bins=bins)

    # add a title and x-axis label
    plt.title("Days to Outcome (Death)")
    plt.xlabel("Days from Prediction Timestamp")
    plt.show()


if __name__ == "__main__":
    plot_days_to_outcome_distribution()
