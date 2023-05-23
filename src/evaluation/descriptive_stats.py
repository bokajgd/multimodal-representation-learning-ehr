from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_SET_DIR = (
    PROJECT_ROOT
    / "data"
    / "feature_sets"
    / "multimodal_rep_learning_ehr_features_2023_05_20_10_28"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval_outputs"

feature_df = pd.read_csv(FEATURE_SET_DIR / "saps_ii_flattened_features.csv")


def _calculate_stats(subset_df: pd.DataFrame, full_df: pd.DataFrame, subset_name: str):
    """Calculate descriptive stats for the given dataframe."""

    # create a dictionary to store the stats
    stats_dict = {}

    # enter subset name to dictionary
    stats_dict["subset_name"] = subset_name

    # enter the number of rows in the subset to the dictionary
    stats_dict["n_icu_stays"] = subset_df.shape[0]

    # enter how big percentage of the full dataset the subset is to the dictionary
    stats_dict["n_icu_stays_percentage"] = (subset_df.shape[0] / full_df.shape[0]) * 100

    # enter the average age of the subset to the dictionary
    stats_dict["age_mean"] = subset_df["pred_age"].mean()

    # enter the interquartile range as an interval of the age of the subset to the dictionary
    stats_dict["age_iqr"] = (
        subset_df["pred_age"].quantile(0.25),
        subset_df["pred_age"].quantile(0.75),
    )

    # enter the number of females in the subset to the dictionary
    stats_dict["n_sex_is_female"] = subset_df["pred_sex_is_female"].sum()

    # enter the percentage of females in the subset to the dictionary
    stats_dict["sex_is_female_percentage"] = (
        subset_df["pred_sex_is_female"].mean() * 100
    )

    # enter the number of patients with a scheduled surgical admission in the subset to the dictionary
    stats_dict["n_scheduled_surgical"] = subset_df[
        "pred_scheduled_surgical_within_1000_days_latest_fallback_0"
    ].sum()

    # enter the percent of patients with an nscheduled surgical admission in the subset to the dictionary
    stats_dict["n_scheduled_surgical_percentage"] = (
        subset_df["pred_scheduled_surgical_within_1000_days_latest_fallback_0"].mean()
        * 100
    )

    # enter the number of patients with an unscheduled surgical admission in the subset to the dictionary
    stats_dict["n_unscheduled_surgical"] = subset_df[
        "pred_unscheduled_surgical_within_1000_days_latest_fallback_0"
    ].sum()

    # enter the percent of patients with an unscheduled surgical admission in the subset to the dictionary
    stats_dict["n_unscheduled_surgical_percentage"] = (
        subset_df["pred_unscheduled_surgical_within_1000_days_latest_fallback_0"].mean()
        * 100
    )

    # enter the number of patients with a medical admission in the subset to the dictionary
    stats_dict["n_medical"] = subset_df[
        "pred_medical_within_1000_days_latest_fallback_0"
    ].sum()

    # enter the percent of patients with a medical admission in the subset to the dictionary
    stats_dict["n_medical_percentage"] = (
        subset_df["pred_medical_within_1000_days_latest_fallback_0"].mean() * 100
    )

    # enter the mean number of ICU stays pr patient of the subset to the dictionary
    if subset_name == "overall":
        stats_dict["n_appearances_pr_patient"] = (
            subset_df.shape[0] / subset_df["patient_id"].nunique()
        )
    else:
        stats_dict["n_appearances_pr_patient"] = None

    return stats_dict


def generate_descriptive_stats_table(
    feature_df: pd.DataFrame = feature_df, save_table: bool = False
) -> pd.DataFrame:
    """Function for generation of descriptive stats table from the study dataset.

    Args:
        eval_df (pd.DataFrame, optional): The evaluation dataset. Defaults to eval_df.

    Returns:
        pd.DataFrame: The descriptive stats table.
    """
    df = feature_df

    # calculate descriptive stats for the no outcome subset
    overall_stats_dict = _calculate_stats(
        subset_df=df,
        full_df=df,
        subset_name="overall",
    )

    # subset rows that have a value of 0 for both ourcome columns
    no_outocome_subset_df = df.loc[
        (df["outc_date_of_death_within_3_days_bool_fallback_0_dichotomous"] == 0)
        & (df["outc_date_of_death_within_30_days_bool_fallback_0_dichotomous"] == 0)
    ]

    # calculate descriptive stats for the no outcome subset
    no_outcome_stats_dict = _calculate_stats(
        subset_df=no_outocome_subset_df,
        full_df=df,
        subset_name="no_outcome",
    )

    # subset rows that have a value of 1 for the 3 day outcome column
    three_day_outcome_subset_df = df.loc[
        (df["outc_date_of_death_within_3_days_bool_fallback_0_dichotomous"] == 1)
    ]

    # calculate descriptive stats for the 3 day outcome subset
    three_day_outcome_stats_dict = _calculate_stats(
        subset_df=three_day_outcome_subset_df,
        full_df=df,
        subset_name="three_day_outcome",
    )

    # subset rows that have a value of 1 for the 30 day outcome column
    thirty_day_outcome_subset_df = df.loc[
        (df["outc_date_of_death_within_30_days_bool_fallback_0_dichotomous"] == 1)
    ]

    # calculate descriptive stats for the 30 day outcome subset
    thirty_day_outcome_stats_dict = _calculate_stats(
        subset_df=thirty_day_outcome_subset_df,
        full_df=df,
        subset_name="thirty_day_outcome",
    )

    # create a list of the stats dictionaries
    stats_dicts = [
        overall_stats_dict,
        no_outcome_stats_dict,
        three_day_outcome_stats_dict,
        thirty_day_outcome_stats_dict,
    ]

    # create a dataframe from the list of stats dictionaries
    stats_df = pd.DataFrame(stats_dicts)

    # round the values in the dataframe to 2 decimals
    stats_df = stats_df.round(3)

    if save_table:
        stats_df.to_csv(
            OUTPUT_DIR / "descriptive_stats_table.csv",
            index=False,
        )

    return stats_df


if __name__ == "__main__":
    generate_descriptive_stats_table()
