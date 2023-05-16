from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

RELATIVE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_SET_PATH = (
    RELATIVE_PROJECT_ROOT
    / "data"
    / "feature_sets"
    / "multimodal_rep_learning_ehr_features_2023_05_15_14_47"
)


def plot_tsne_projections():
    # Load data
    train_df = pd.read_csv(
        FEATURE_SET_PATH
        / "flattened_13535rows_567cols_train_admission_level_vectors_df.csv",
        index_col=False,
    )

    # Extract only 500 rows from the train set
    train_df = train_df.sample(n=3000, random_state=0)

    # Calculate t-SNE projections
    tsne_model = TSNE(
        n_components=2,
        perplexity=100,
        learning_rate=10,
        n_iter=2500,
        verbose=1,
        random_state=0,
        angle=0.75,
    )
    tsne_projections = tsne_model.fit_transform(train_df.iloc[:, 9:])

    # Create colourmap
    binary_colour_map = np.array(["#104547", "#C13B3B"])

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=tsne_projections[:, 0],
        y=tsne_projections[:, 1],
        hue="outc_date_of_death_within_30_days_bool_fallback_0_dichotomous",
        palette=sns.color_palette(binary_colour_map, 2),
        data=train_df,
        legend="full",
        alpha=0.3,
    ).set_title("tSNE Projections coloured by outcome label")

    plt.show()

    print("Done")


def plot_pca_projections():
    # Load data
    train_df = pd.read_csv(
        FEATURE_SET_PATH
        / "flattened_13535rows_567cols_train_admission_level_vectors_df.csv",
        index_col=False,
    )

    # Extract only 500 rows from the train set
    train_df = train_df.sample(n=3000, random_state=1)

    # Calculate PCA components
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(train_df.iloc[:, 9:].values)

    print(
        "Explained variation per principal component: {}".format(
            pca.explained_variance_ratio_
        )
    )

    # Create colourmap
    binary_colour_map = np.array(["#104547", "#C13B3B"])

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        hue="outc_date_of_death_within_30_days_bool_fallback_0_dichotomous",
        palette=sns.color_palette(binary_colour_map, 2),
        data=train_df,
        legend="full",
        alpha=0.3,
    ).set_title("PCA projections coloured by outcome label")

    plt.show()

    print("Done")


if "__main__" == __name__:
    plot_pca_projections()
