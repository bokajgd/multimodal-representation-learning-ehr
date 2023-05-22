from loaders.load_notes import load_notes
from text_model_pipeline import text_model_pipeline
from utils.utils import DATA_PATH

import pandas as pd


def load_notes_for_text_model() -> pd.DataFrame:
    """Loads the notes for the text model pipeline.
    Keeps only notes that are not within any observation window.

    Returns:
        pd.DataFrame: The notes dataframes"""

    df = load_notes(nrows=500000)

    predictions_times_df_path = DATA_PATH / "misc" / "cohort_with_prediction_times.csv"
    prediction_times_df = pd.read_csv(predictions_times_df_path)

    # only keep notes that are not within any observation window
    prediction_times_df = prediction_times_df.sort_values(["patient_id", "timestamp"])

    for _, row in prediction_times_df.iterrows():
        patient_id = row["patient_id"]
        prediction_timestamp = row["timestamp"]

        patient_data = df[df["patient_id"] == patient_id]
        filtered_data = patient_data[
            (
                pd.to_datetime(patient_data["timestamp"])
                >= pd.to_datetime(prediction_timestamp)
            )
            | (
                pd.to_datetime(patient_data["timestamp"])
                < pd.to_datetime(prediction_timestamp) - pd.Timedelta(hours=24)
            )
        ]

        df = df[df["patient_id"] != patient_id]
        df = pd.concat([df, filtered_data])

    return df


if __name__ == "__main__":
    df = load_notes_for_text_model()

    text_model_pipeline(
        model="tfidf",
        df=df,
        max_features=500,
        max_df=0.90,
        min_df=10,
        ngram_range=(1, 3),
    )
    print("Done fitting and saving text models.")
