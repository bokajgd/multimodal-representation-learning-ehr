"""Full feature set specification module."""
import logging
import sys

sys.path.append(".")

from typing import Union

import numpy as np
from psycop_feature_generation.application_modules.project_setup import ProjectInfo
from text_features.loaders.load_notes import load_notes
from timeseriesflattener.feature_spec_objects import (
    BaseModel,
    OutcomeGroupSpec,
    OutcomeSpec,
    PredictorGroupSpec,
    PredictorSpec,
    StaticSpec,
    TextPredictorSpec,
    _AnySpec,
)
from timeseriesflattener.text_embedding_functions import sklearn_embedding

from .utils import load_text_model


class SpecSet(BaseModel):
    """A set of unresolved specs, ready for resolving."""

    temporal_predictors: list[PredictorSpec]
    static_predictors: list[StaticSpec]


class FullFeatureSpecifier:
    def __init__(
        self,
        project_info: ProjectInfo,
        min_set_for_debug: bool = False,
        get_text_specs: bool = True,
    ):
        self.min_set_for_debug = min_set_for_debug
        self.project_info = project_info
        self.get_text_specs = get_text_specs

    def _get_metadata_specs(self) -> list[_AnySpec]:
        """Get metadata specs for evalutation."""
        print("-------- Generating metadata specs --------")

        admission_types = PredictorGroupSpec(
            values_loader=("scheduled_surgical", "unscheduled_surgical", "medical"),
            lookbehind_days=[1000],
            resolve_multiple_fn=["latest"],
            fallback=[0],
            allowed_nan_value_prop=[0],
            prefix=self.project_info.prefix.eval,
        ).create_combinations()

        return admission_types

    def _get_outcome_specs(self) -> list[OutcomeSpec]:
        """Get outcome specs."""
        print("-------- Generating outcome specs --------")

        if self.min_set_for_debug:
            return [
                OutcomeSpec(
                    values_loader="date_of_death",
                    lookahead_days=30,
                    resolve_multiple_fn="bool",
                    fallback=0,
                    incident=True,
                    allowed_nan_value_prop=0,
                    prefix=self.project_info.prefix.outcome,
                ),
            ]

        return OutcomeGroupSpec(
            values_loader=["date_of_death"],
            lookahead_days=[3, 30],
            resolve_multiple_fn=["bool"],
            fallback=[0],
            incident=[True],
            allowed_nan_value_prop=[0],
            prefix=self.project_info.prefix.outcome,
        ).create_combinations()

    def _get_static_predictor_specs(self):
        """Get static predictor specs."""

        print("-------- Generating static specs --------")

        return [
            StaticSpec(
                values_loader="sex_is_female",
                input_col_name_override="sex_is_female",
                prefix=self.project_info.prefix.predictor,
            ),
        ]

    def _get_admissions_specs(
        self,
        resolve_multiple=None,
        interval_days=None,
    ):
        """Get admissions specs."""

        print("–––––––– Generating admissions specs ––––––––")

        admissions = PredictorGroupSpec(
            values_loader=("scheduled_surgical", "unscheduled_surgical", "medical"),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[0],
            allowed_nan_value_prop=[0],
        ).create_combinations()

        return admissions

    def _get_emergency_admissions_specs(
        self,
        resolve_multiple=None,
        interval_days=None,
    ):
        """Get emergency admissions specs."""

        print("–––––––– Generating emergency admissions specs ––––––––")

        emergency_admissions = PredictorGroupSpec(
            values_loader=("emergency_admissions",),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[np.NaN],
            allowed_nan_value_prop=[0],
        ).create_combinations()

        return emergency_admissions

    def _get_inputevents_specs(
        self,
        resolve_multiple=None,
        interval_days=None,
    ):
        """Get inputevents specs."""

        print("-------- Generating inputevents specs --------")

        inputevents = PredictorGroupSpec(
            values_loader=(
                "nacl_0_9_ml",
                "gastric_meds_ml",
                "potassium_chloride_meq",
                "sterile_water_ml",
                "fentanyl_mg",
                "fresh_frozen_plasma_ml",
                "albumin_5_ml",
            ),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[np.NaN],
            allowed_nan_value_prop=[0],
        ).create_combinations()

        return inputevents

    def _get_weight_specs(
        self,
        resolve_multiple=None,
        interval_days=None,
    ):
        """Get weight specs."""

        print("-------- Generating weight specs --------")

        weight = PredictorGroupSpec(
            values_loader=("weight",),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[np.NaN],
            allowed_nan_value_prop=[0],
        ).create_combinations()

        return weight

    def _get_diagnoses_specs(
        self,
        resolve_multiple=None,
        interval_days=None,
    ):
        """Get diagnoses specs."""

        print("–––––––– Generating diagnoses specs ––––––––")

        diagnoses = PredictorGroupSpec(
            values_loader=(
                "metastatic_cancer",
                "hematologic_malignancy",
                "acquired_immunodeficiency_syndrome",
            ),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[0],
            allowed_nan_value_prop=[0],
        ).create_combinations()

        return diagnoses

    def _get_chartevents_specs(
        self,
        resolve_multiple=None,
        interval_days=None,
    ):
        """Get chartevents specs."""

        print("–––––––– Generating chartevents specs ––––––––")

        chartevents = PredictorGroupSpec(
            values_loader=(
                "systolic_blood_pressure",
                "heart_rate",
                "gcs",
                "pao2_fio2_ratio",
                "temperature",
            ),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[np.NaN],
            allowed_nan_value_prop=[0],
        ).create_combinations()

        return chartevents

    def _get_labevents_specs(
        self,
        resolve_multiple=None,
        interval_days=None,
    ):
        """Get labevents specs."""

        print("–––––––– Generating labevents specs ––––––––")

        labevents = PredictorGroupSpec(
            values_loader=(
                "white_blod_cells",
                "sodium_level",
                "potassium_level",
                "bicarbonate",
                "bilirubin_level",
                "urea_nitrogen",
            ),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[np.NaN],
            allowed_nan_value_prop=[0],
        ).create_combinations()

        return labevents

    def _get_outputevents_specs(
        self,
        resolve_multiple=None,
        interval_days=None,
    ):
        """Get outputevents specs."""

        print("–––––––– Generating outputevents specs ––––––––")

        outputevents = PredictorGroupSpec(
            values_loader=("urine",),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[np.NaN],
            allowed_nan_value_prop=[0],
        ).create_combinations()

        return outputevents

    def _get_tfidf_all_notes_specs(
        self,
        resolve_multiple=None,
        interval_days=None,
    ):
        """Get specs for tfidf features from all notes."""

        print("–––––––– Generating tfidf specs ––––––––")

        tfidf_model = load_text_model(
            filename="tfidf_ngram_range_13_max_df_066_min_df_10_max_features_500.pkl",
        )

        tfidf = TextPredictorSpec(
            values_loader=load_notes,
            lookbehind_days=interval_days,
            fallback=0,
            resolve_multiple_fn=resolve_multiple,
            feature_name="text_tfidf",
            interval_days=interval_days,
            input_col_name_override="text",
            embedding_fn=sklearn_embedding,
            embedding_fn_kwargs={"model": tfidf_model},
        )

        return tfidf

    def _get_temporal_predictor_specs(self) -> list[PredictorSpec]:
        """Generate predictor spec list."""
        print("–––––––– Generating temporal predictor specs ––––––––")

        if self.min_set_for_debug:
            return [
                PredictorSpec(
                    values_loader="weight",
                    lookbehind_days=2,
                    resolve_multiple_fn="change_per_day",
                    fallback=np.NaN,
                    allowed_nan_value_prop=0,
                    prefix=self.project_info.prefix.predictor,
                ),
            ]

        admissions = self._get_admissions_specs(
            resolve_multiple=["latest"],
            interval_days=[1000],
        )

        emergency_admissions = self._get_emergency_admissions_specs(
            resolve_multiple=["sum", "count"],
            interval_days=[30, 182],
        )

        diagnoses = self._get_diagnoses_specs(
            resolve_multiple=["bool"],
            interval_days=[1000],
        )

        inputevents = self._get_inputevents_specs(
            resolve_multiple=["sum", "latest"],
            interval_days=[0.5, 1],
        )

        weight = self._get_weight_specs(
            resolve_multiple=["min", "max", "mean", "latest", "change_per_day"],
            interval_days=[1, 30, 182],
        )

        chartevents = self._get_chartevents_specs(
            resolve_multiple=["min", "max", "mean", "latest"],
            interval_days=[1, 30, 182],
        )

        labevents = self._get_labevents_specs(
            resolve_multiple=["min", "max", "mean", "latest"],
            interval_days=[1, 30, 182],
        )

        outputevents = self._get_outputevents_specs(
            resolve_multiple=["sum"],
            interval_days=[0.5, 1],
        )

        return (
            admissions
            + emergency_admissions
            + diagnoses
            + inputevents
            + weight
            + chartevents
            + labevents
            + outputevents
        )

    def _get_text_predictor_specs(self) -> list[TextPredictorSpec]:
        """Generate text predictor spec list."""

        print("–––––––– Generating text predictor specs ––––––––")

        noteevents = self._get_tfidf_all_notes_specs(
            resolve_multiple="concatenate",
            interval_days=2,
        )

        return [noteevents]

    def get_feature_specs(self) -> list[Union[TextPredictorSpec, PredictorSpec]]:
        """Get a spec set."""

        if self.min_set_for_debug:
            return (
                self._get_temporal_predictor_specs()
                + self._get_static_predictor_specs()
                + self._get_outcome_specs()
                + self._get_metadata_specs()
            )

        print("–––––––– Done generating specs ––––––––")

        if self.get_text_specs:
            return (
                self._get_temporal_predictor_specs()
                + self._get_text_predictor_specs()
                + self._get_static_predictor_specs()
                + self._get_outcome_specs()
                + self._get_metadata_specs()
            )
        else:
            return (
                self._get_temporal_predictor_specs()
                + self._get_static_predictor_specs()
                + self._get_outcome_specs()
                + self._get_metadata_specs()
            )
