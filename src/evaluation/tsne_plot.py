from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

RELATIVE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_SET_PATH = (
    RELATIVE_PROJECT_ROOT
    / "data"
    / "feature_sets"
    / "multimodal_rep_learning_ehr_features_2023_05_17_02_15"
)
PLOT_OUTPUT_PATH = RELATIVE_PROJECT_ROOT / "outputs" / "eval_outputs"


def _calculate_tsne_components(
    df: pd.DataFrame,
    n_samples: Optional[int] = None,
    use_cols_from: int = 0,
    use_cols_to: int = 850,
    n_components: int = 2,
    perplexity: int = 50,
    learning_rate: int = 100,
    n_iter: int = 2500,
    random_state: int = 0,
    angle: float = 0.5,
):
    # impute all nan values with 0
    df = df.fillna(0)

    if n_samples is not None:
        df = df.sample(n=n_samples, random_state=0)

    # Calculate t-SNE projections
    tsne_model = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        verbose=1,
        random_state=random_state,
        angle=angle,
    )

    tsne_projections = tsne_model.fit_transform(df.iloc[:, use_cols_from:use_cols_to])

    comp_1 = tsne_projections[:, 0]
    comp_2 = tsne_projections[:, 1]

    return comp_1, comp_2


def _plot_tsne_projections(
    df: pd.DataFrame,
    comp_1: np.ndarray,
    comp_2: np.ndarray,
    colour_by_column: str = "outc_date_of_death_within_30_days_bool_fallback_0_dichotomous",
    colour_by_column_name: str = "outcome label",
    save_plot: bool = False,
    indices_to_plot: Optional[list] = None,
):
    """
    Plot t-SNE projections

    Args:
        df (pd.DataFrame): DataFrame containing the data to project and metadata columns to colour the points by
        comp_1 (np.ndarray): First component of the t-SNE projections
        comp_2 (np.ndarray): Second component of the t-SNE projections
        colour_by_column (str): Name of the column to colour the points by
        colour_by_column_name (str): Name of the column to colour the points by
        save_plot (bool): Whether to save the plot to disk
        indices_to_plot (list): List of indices to plot

    Returns:
        None
    """

    if indices_to_plot is not None:
        comp_1 = comp_1[indices_to_plot]
        comp_2 = comp_2[indices_to_plot]
        df = df.iloc[indices_to_plot]

    # if number of unique values in colour_by_column is 2, use a binary colourmap
    if len(df[colour_by_column].unique()) == 2:
        colour_map = np.array(["#104547", "#C13B3B"])
    else:
        # set a colour map with the number of colours equal to the number of unique values in colour_by_column
        colour_map = sns.color_palette("hls", len(df[colour_by_column].unique()))

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=comp_1,
        y=comp_2,
        hue=colour_by_column,
        palette=sns.color_palette(colour_map, len(df[colour_by_column].unique())),
        data=df,
        legend="full",
        alpha=0.3,
    ).set_title(f"tSNE projections coloured by {colour_by_column_name}")

    if save_plot:
        plt.savefig(
            PLOT_OUTPUT_PATH / f"tsne_projections_coloured_by_{colour_by_column}.png",
            dpi=1000,
        )

    plt.show()

    print("t-SNE projections plotted")


def calculate_feature_tsne_compoents(
    n_components: int = 2,
    perplexity: int = 50,
    learning_rate: int = 100,
    n_iter: int = 2500,
    random_state: int = 0,
    angle: float = 0.5,
):
    """
    Calculate t-SNE projections of feature vectors
    """

    # Load data
    df = pd.read_csv(
        FEATURE_SET_PATH / "full_with_text_features_850rows_850cols_co_counts.csv",
        index_col=False,
    )

    comp_1, comp_2 = _calculate_tsne_components(
        df,
        use_cols_from=0,
        use_cols_to=850,
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=random_state,
        angle=angle,
    )

    return df, comp_1, comp_2


def plot_tsne_feature_vectors_by_feature_type(
    df: pd.DataFrame, comp_1: np.ndarray, comp_2: np.ndarray
):
    """
    Plot t-SNE projections of feature vectors coloured by feature type

    Args:
        df (pd.DataFrame): DataFrame containing the data to project and metadata columns to colour the points by
        comp_1 (np.ndarray): First component of the t-SNE projections
        comp_2 (np.ndarray): Second component of the t-SNE projections
    """

    # Add a column labelling cooccurrence vectors that are text features
    df["is_text_feature"] = "No"

    for col in df.columns:
        if "text" in col.lower():
            index = df.columns.get_loc(col)
            df.loc[index, "is_text_feature"] = "Yes"

    # Remove last row
    df = df.iloc[:-1, :]

    # Plot t-SNE projections
    _plot_tsne_projections(
        df=df,
        comp_1=comp_1,
        comp_2=comp_2,
        colour_by_column="is_text_feature",
        colour_by_column_name="feature type (text or non-text)",
        save_plot=False,
    )


def plot_tsne_feature_vectors_by_quantile(
    df: pd.DataFrame, comp_1: np.ndarray, comp_2: np.ndarray
):
    """
    Plot t-SNE projections of feature vectors coloured by quantile
    """

    # Add a column labelling which quantile the feature represents (if numeric)
    df["feature_quantile"] = "Non numeric"

    for col in df.columns:
        if "p15" in col.lower():
            index = df.columns.get_loc(col)
            df.loc[index, "feature_quantile"] = "Lower 15%"
        elif "p_mid" in col.lower():
            index = df.columns.get_loc(col)
            df.loc[index, "feature_quantile"] = "Middle 70%"
        elif "p85" in col.lower():
            index = df.columns.get_loc(col)
            df.loc[index, "feature_quantile"] = "Upper 15%"

    # Get indices for columns with Pao2/Fio2 ratio features
    pao2_fio_2_ration_indices = [
        idx for idx, column in enumerate(df.columns) if "pao2_fio2" in column.lower()
    ]

    # Plot t-SNE projections
    _plot_tsne_projections(
        df=df,
        comp_1=comp_1,
        comp_2=comp_2,
        colour_by_column="feature_quantile",
        colour_by_column_name="feature quantile category for all PaO2/FiO2 related features)",
        save_plot=False,
        indices_to_plot=pao2_fio_2_ration_indices,
    )

    print("Done")


if "__main__" == __name__:
    df, comp_1, comp_2 = calculate_feature_tsne_compoents()
    plot_tsne_feature_vectors_by_feature_type(df, comp_1, comp_2)
    plot_tsne_feature_vectors_by_quantile(df, comp_1, comp_2)
