"""Function for extracting vocabulary from a tfidf model."""

import pickle
from pathlib import Path

RELATIVE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
VOCAB_PICKLE_PATH = (
    RELATIVE_PROJECT_ROOT
    / "text_models"
    / "vocabs"
    / "tfidf_ngram_range_13_max_df_066_min_df_10_max_features_500.pkl_vocab.pkl"
)

DATA_OUTPUT_PATH = RELATIVE_PROJECT_ROOT / "data" / "misc"


def get_vocab():
    """Function for extracting vocabulary from a tfidf model."""

    # open the pickle file
    with open(VOCAB_PICKLE_PATH, "rb") as f:
        vocab = pickle.load(f)

    # save terms to a text file
    with open(DATA_OUTPUT_PATH / "tfidf_vocab.txt", "w") as f:
        for word in vocab:
            f.write(word + "\n")

    return vocab


if __name__ == "__main__":
    vocab = get_vocab()
