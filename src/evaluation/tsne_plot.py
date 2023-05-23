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
EVAL_DF_PATH = (
    RELATIVE_PROJECT_ROOT
    / "outputs"
    / "model_outputs"
    / "model_outputs_05_20_2023_13_12_18"
)


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
    custom_legend_title: str,
    custom_legend_labels: Optional[list] = None,
    colour_by_column: str = "outc_date_of_death_within_30_days_bool_fallback_0_dichotomous",
    colour_by_column_name: str = "outcome label",
    save_plot: bool = False,
    indices_to_plot: Optional[list] = None,
):
    """Plot t-SNE projections.

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

    # adapt the size of points based on the number of points
    default_point_size = 50

    num_points = len(df)
    if num_points < 25:
        point_size = default_point_size * 3
    else:
        point_size = default_point_size

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=comp_1,
        y=comp_2,
        hue=colour_by_column,
        palette=sns.color_palette(colour_map, len(df[colour_by_column].unique())),
        data=df,
        legend="full",
        alpha=0.3,
        s=point_size,
    ).set_title(f"tSNE projections coloured by {colour_by_column_name}")

    if custom_legend_title is not None:
        legend_title = custom_legend_title
        label_names = custom_legend_labels

        ax = plt.gca()

        handles, labels = ax.get_legend_handles_labels()

        new_legend = plt.legend(handles, label_names, title=legend_title)

        ax.add_artist(new_legend)

    if save_plot:
        plt.savefig(
            PLOT_OUTPUT_PATH / f"tsne_projections_coloured_by_{colour_by_column}.png",
            dpi=1000,
        )

    plt.show()

    print("t-SNE projections plotted")


def calculate_co_vectors_tsne_compoents(
    n_components: int = 2,
    perplexity: int = 25,
    learning_rate: int = 200,
    n_iter: int = 2000,
    random_state: int = 0,
    angle: float = 0.5,
):
    """Calculate t-SNE projections of feature vectors."""

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


def calculate_patient_embeddings_tsne_compoents(
    n_components: int = 2,
    perplexity: int = 50,
    learning_rate: int = 200,
    n_iter: int = 2000,
    random_state: int = 0,
    angle: float = 0.5,
):
    """Calculate t-SNE projections of feature vectors."""

    # Load data
    df = pd.read_csv(
        FEATURE_SET_PATH
        / "full_with_text_features_13535rows_860cols_train_admission_level_vectors_df.csv",
        index_col=False,
    )

    # add admissions_type column where 0 = unscheduled_surgical, 1 = scheduled_surgical, 2 = medical, 3 = other
    flattened_df = pd.read_csv(
        FEATURE_SET_PATH
        / "full_with_text_features_13535rows_631cols_train_flattened_features.csv",
        index_col=False,
    )

    flattened_df = flattened_df.rename(
        columns={
            "pred_unscheduled_surgical_within_2_days_latest_fallback_0": "unscheduled_surgical",
            "pred_scheduled_surgical_within_2_days_latest_fallback_0": "scheduled_surgical",
            "pred_medical_within_2_days_latest_fallback_0": "medical",
        }
    )

    df = df.merge(
        flattened_df[
            [
                "unscheduled_surgical",
                "scheduled_surgical",
                "medical",
            ]
        ],
        left_index=True,
        right_index=True,
    )

    df["admissions_type"] = df.apply(
        lambda row: 0
        if row["unscheduled_surgical"] == 1
        else 1
        if row["scheduled_surgical"] == 1
        else 2
        if row["medical"] == 1
        else 3,
        axis=1,
    )

    comp_1, comp_2 = _calculate_tsne_components(
        df,
        use_cols_from=10,
        use_cols_to=860,
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=random_state,
        angle=angle,
    )

    return df, comp_1, comp_2


def plot_tsne_patient_ebmeddings_by_outcome_label(
    df: pd.DataFrame,
    comp_1: np.ndarray,
    comp_2: np.ndarray,
    outcome_prediction_window: int = 30,
    save_plot: bool = False,
):
    """
    Plot t-SNE projections of patient embeddings coloured by outcome label.

    Args:
        df (pd.DataFrame): DataFrame containing the data to project and metadata columns to colour the points by
        comp_1 (np.ndarray): First component of the t-SNE projections
        comp_2 (np.ndarray): Second component of the t-SNE projections
        save_plot (bool): Whether to save the plot to disk
    """

    # Plot t-SNE projections coloured by outcome label
    _plot_tsne_projections(
        df=df,
        comp_1=comp_1,
        comp_2=comp_2,
        custom_legend_title="Outcome label",
        custom_legend_labels=(
            "No death within 30 days of admission",
            "Death within 30 days of admission",
        ),
        colour_by_column=f"outc_date_of_death_within_{outcome_prediction_window}_days_bool_fallback_0_dichotomous",
        colour_by_column_name="outcome label",
        save_plot=save_plot,
    )


def plot_tsne_patient_ebmeddings_by_age_bin(
    df: pd.DataFrame,
    comp_1: np.ndarray,
    comp_2: np.ndarray,
    save_plot: bool = False,
):
    """
    Plot t-SNE projections of patient embeddings coloured by age bin.

    Args:
        df (pd.DataFrame): DataFrame containing the data to project and metadata columns to colour the points by
        comp_1 (np.ndarray): First component of the t-SNE projections
        comp_2 (np.ndarray): Second component of the t-SNE projections
        save_plot (bool): Whether to save the plot to disk
    """

    # create column that bins 'pred_age' into 10 year bins
    df["age_bins"] = pd.cut(
        df["age"],
        bins=[10, 20, 30, 40, 50, 60, 70, 80, 90],
        labels=["10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", ">80"],
    )

    # Plot t-SNE projections coloured by outcome label
    _plot_tsne_projections(
        df=df,
        comp_1=comp_1,
        comp_2=comp_2,
        custom_legend_title="Age",
        custom_legend_labels=(
            "10-20",
            "20-30",
            "30-40",
            "40-50",
            "50-60",
            "60-70",
            "70-80",
            ">80",
        ),
        colour_by_column="age_bins",
        colour_by_column_name="age category",
        save_plot=save_plot,
    )


def plot_tsne_patient_ebmeddings_by_admission_type(
    df: pd.DataFrame,
    comp_1: np.ndarray,
    comp_2: np.ndarray,
    save_plot: bool = False,
):
    """
    Plot t-SNE projections of patient embeddings coloured by admission type.

    Args:
        df (pd.DataFrame): DataFrame containing the data to project and metadata columns to colour the points by
        comp_1 (np.ndarray): First component of the t-SNE projections
        comp_2 (np.ndarray): Second component of the t-SNE projections
        save_plot (bool): Whether to save the plot to disk
    """

    # Plot t-SNE projections coloured by outcome label
    _plot_tsne_projections(
        df=df,
        comp_1=comp_1,
        comp_2=comp_2,
        custom_legend_title="Admission type",
        custom_legend_labels=(
            "Unscheduled surgical",
            "Scheduled surgical",
            "Medical",
        ),
        colour_by_column="admissions_type",
        colour_by_column_name="admission type",
        save_plot=save_plot,
    )


def plot_tsne_co_vectors_by_feature_type(
    df: pd.DataFrame,
    comp_1: np.ndarray,
    comp_2: np.ndarray,
    save_plot: bool = False,
):
    """Plot t-SNE projections of feature vectors coloured by feature type."""

    # Add a column labelling cooccurrence vectors that are text features
    df["is_text_feature"] = 0

    for col in df.columns:
        if "text" in col.lower():
            index = df.columns.get_loc(col)
            df.loc[index, "is_text_feature"] = 1

    # Remove last row
    df = df.iloc[:-1, :]

    # Plot t-SNE projections
    _plot_tsne_projections(
        df=df,
        comp_1=comp_1,
        comp_2=comp_2,
        custom_legend_title="Feature type",
        custom_legend_labels=(
            "Non-text feature",
            "Text feature",
        ),
        colour_by_column="is_text_feature",
        colour_by_column_name="feature type (text or non-text)",
        save_plot=save_plot,
    )

    print("Done")


def plot_tsne_co_vectors_by_quantile(
    df: pd.DataFrame,
    comp_1: np.ndarray,
    comp_2: np.ndarray,
    save_plot: bool = False,
):
    """Plot t-SNE projections of feature vectors coloured by quantile."""

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

    # Remove last row
    df = df.iloc[:-1, :]

    # Get indices of rows that are numeric features
    numeric_features_indices = df[df["feature_quantile"] != "Non numeric"].index

    # Plot t-SNE projections
    _plot_tsne_projections(
        df=df,
        comp_1=comp_1,
        comp_2=comp_2,
        custom_legend_title="Feature quantile category",
        custom_legend_labels=(
            "Lower 15%",
            "Middle 70%",
            "Upper 15%",
        ),
        colour_by_column="feature_quantile",
        colour_by_column_name="feature quantile category for all numeric features)",
        save_plot=save_plot,
        indices_to_plot=numeric_features_indices,
    )

    print("Done")


def plot_tsne_pao2_fio_2_co_vectors_by_quantile(
    df: pd.DataFrame,
    comp_1: np.ndarray,
    comp_2: np.ndarray,
    save_plot: bool = False,
):
    """Plot t-SNE projections of feature vectors coloured by quantile."""

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

    # Remove last row
    df = df.iloc[:-1, :]

    # Get indices for columns with Pao2/Fio2 ratio features
    pao2_fio_2_ration_indices = [
        idx for idx, column in enumerate(df.columns) if "pao2_fio2" in column.lower()
    ]

    # Plot t-SNE projections
    _plot_tsne_projections(
        df=df,
        comp_1=comp_1,
        comp_2=comp_2,
        custom_legend_title="Feature quantile category",
        custom_legend_labels=(
            "Lower 15%",
            "Middle 70%",
            "Upper 15%",
        ),
        colour_by_column="feature_quantile",
        colour_by_column_name="feature quantile category for all PaO2/FiO2 related features)",
        save_plot=save_plot,
        indices_to_plot=pao2_fio_2_ration_indices,
    )

    print("Done")


def plot_tsne_heart_rate_blood_pressure_feature_co_vectors(
    df: pd.DataFrame,
    comp_1: np.ndarray,
    comp_2: np.ndarray,
    save_plot: bool = False,
):
    """Plot t-SNE projections of feature vectors coloured by quantile."""

    # Add a column labelling whether the feature is a heart rate or blood pressure feature
    df["heart_rate_or_blood_pressure"] = 0

    for col in df.columns:
        if "heart_rate" in col.lower():
            index = df.columns.get_loc(col)
            df.loc[index, "heart_rate_or_blood_pressure"] = 1
        elif "blood_pressure" in col.lower():
            index = df.columns.get_loc(col)
            df.loc[index, "heart_rate_or_blood_pressure"] = 2

    # remove last two rows
    df = df.iloc[:-2, :]

    # get indices of rows where feature_quantile is "Middle 70%"'
    middle_70_indices = df[df["feature_quantile"] == "Middle 70%"].index

    # Plot t-SNE projections
    _plot_tsne_projections(
        df=df,
        comp_1=comp_1,
        comp_2=comp_2,
        custom_legend_title="Feature",
        custom_legend_labels=(
            "Other",
            "Heart rate",
            "Systolic Blood pressure",
        ),
        colour_by_column="heart_rate_or_blood_pressure",
        colour_by_column_name="whether the feature is a heart rate or systolic blood pressure feature or neither",
        save_plot=save_plot,
        indices_to_plot=middle_70_indices,
    )

    print("Done")


if "__main__" == __name__:
    (
        patient_df,
        patient_comp1,
        patient_comp2,
    ) = calculate_patient_embeddings_tsne_compoents()
    plot_tsne_patient_ebmeddings_by_admission_type(
        patient_df, patient_comp1, patient_comp2
    )
    plot_tsne_patient_ebmeddings_by_age_bin(patient_df, patient_comp1, patient_comp2)
    plot_tsne_patient_ebmeddings_by_outcome_label(
        patient_df, patient_comp1, patient_comp2
    )
    co_df, co_comp_1, co_comp_2 = calculate_co_vectors_tsne_compoents()
    plot_tsne_co_vectors_by_feature_type(co_df, co_comp_1, co_comp_2)
    plot_tsne_co_vectors_by_quantile(co_df, co_comp_1, co_comp_2)
    plot_tsne_pao2_fio_2_co_vectors_by_quantile(co_df, co_comp_1, co_comp_2)
    plot_tsne_heart_rate_blood_pressure_feature_co_vectors(co_df, co_comp_1, co_comp_2)

    print("Done")
