"""Script for misc.

plots and tables.
"""
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DF_PATH = (
    PROJECT_ROOT / "outputs" / "model_outputs" / "model_outputs_05_13_2023_00_41_54"
)
eval_df = pd.read_csv(EVAL_DF_PATH / "evaluation_dataset.csv")


def plot_days_to_outcome_distribution(
    eval_df: pd.DataFrame = eval_df,
    save_plot: bool = False,
) -> None:
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

    plt.figure(figsize=(12, 6))
    sns.histplot(data=eval_df, x="days_to_outcome", bins=bins, color="#ff91a2")

    # add a title and x-axis label
    plt.title("Days to Outcome (Death)")
    plt.xlabel("Days from Prediction Timestamp")

    if save_plot:
        plt.savefig(
            PROJECT_ROOT
            / "outputs"
            / "eval_outputs"
            / "days_to_outcome_distribution.png",
        )

    plt.show()

    print("Done")


def plot_outcome_labe_pie_chart(
    eval_df: pd.DataFrame = eval_df,
    save_plot: bool = False,
) -> None:
    """Plot a pie chart showing the fraction of positive and negative outcome
    labels.

    Args:
        eval_df (pd.DataFrame, optional): The evaluation dataset. Defaults to eval_df.

    Returns:
        None
    """

    labels = ["Death not recorded", "Death recorded"]
    colors = ["#a9d1a7", "#ff91a2"]

    label_counts = eval_df["eval_dod"].notnull().value_counts()

    plt.figure(figsize=(6, 6))
    plt.title("Distribution of patients with and without recorded death")

    plt.pie(label_counts, colors=colors, autopct="%1.1f%%", startangle=90)

    # Create a legend with the labels
    plt.legend(labels=labels, loc="lower left", bbox_to_anchor=(-0.1, -0.1))

    plt.axis("equal")

    if save_plot:
        plt.savefig(
            PROJECT_ROOT / "outputs" / "eval_outputs" / "outcome_label_pie_chart.png",
        )

    plt.show()

    print("Done")


if __name__ == "__main__":
    plot_days_to_outcome_distribution()
    plot_outcome_labe_pie_chart()
