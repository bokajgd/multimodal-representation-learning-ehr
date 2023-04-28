"""Feature specification module."""
import logging
import sys

sys.path.append(".")

from typing import Union

import numpy as np
from psycop_feature_generation.application_modules.project_setup import ProjectInfo
from text_features.loaders.load_notes import load_notes
from timeseriesflattener.feature_spec_objects import (
    BaseModel,
    PredictorGroupSpec,
    PredictorSpec,
    StaticSpec,
    TextPredictorSpec,
    _AnySpec,
)
from timeseriesflattener.text_embedding_functions import sklearn_embedding

from .utils import load_text_model

log = logging.getLogger(__name__)


class SpecSet(BaseModel):
    """A set of unresolved specs, ready for resolving."""

    temporal_predictors: list[PredictorSpec]
    static_predictors: list[StaticSpec]


class FeatureSpecifier:
    def __init__(self, project_info: ProjectInfo, min_set_for_debug: bool = False):
        self.min_set_for_debug = min_set_for_debug
        self.project_info = project_info

    def _get_static_predictor_specs(self):
        """Get static predictor specs."""
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

        log.info("–––––––– Generating admissions specs ––––––––")

        admissions = PredictorGroupSpec(
            values_loader=("emergency_admissions",),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[0],
            allowed_nan_value_prop=[0],
        ).create_combinations()

        return admissions

    def _get_inputevents_specs(
        self,
        resolve_multiple=None,
        interval_days=None,
    ):
        """Get inputevents specs."""

        log.info("–––––––– Generating inputevents specs ––––––––")

        inputevents = PredictorGroupSpec(
            values_loader=("nacl_0_9_ml", "weight", "fentanyl_mg"),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[0],
            allowed_nan_value_prop=[0],
        ).create_combinations()

        return inputevents

    def _get_chartevents_specs(
        self,
        resolve_multiple=None,
        interval_days=None,
    ):
        """Get chartevents specs."""

        log.info("–––––––– Generating chartevents specs ––––––––")

        admissions = PredictorGroupSpec(
            values_loader=("gcs",),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[0],
            allowed_nan_value_prop=[0],
        ).create_combinations()

        return admissions

    def _get_tfidf_all_notes_specs(
        self,
        resolve_multiple=None,
        interval_days=None,
    ):
        """Get specs for tfidf features from all notes."""

        log.info("–––––––– Generating tfidf specs ––––––––")

        tfidf_model = load_text_model(
            filename="tfidf_ngram_range_13_max_df_095_min_df_10_max_features_250.pkl",
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
        log.info("–––––––– Generating temporal predictor specs ––––––––")

        if self.min_set_for_debug:
            return [
                PredictorSpec(
                    values_loader="weight",
                    lookbehind_days=2,
                    resolve_multiple_fn="latest",
                    fallback=0,
                    allowed_nan_value_prop=0,
                    prefix=self.project_info.prefix.predictor,
                ),
            ]

        admissions = self._get_admissions_specs(
            resolve_multiple=["count", "mean"],
            interval_days=[182, 365],
        )

        inputevents = self._get_inputevents_specs(
            resolve_multiple=["max", "min", "mean"],
            interval_days=[0.146, 2, 30],
        )

        chartevents = self._get_chartevents_specs(
            resolve_multiple=["latest", "mean", "change_per_day"],
            interval_days=[2],
        )

        return admissions + inputevents + chartevents

    def _get_text_predictor_specs(self) -> list[TextPredictorSpec]:
        """Generate text predictor spec list."""

        log.info("–––––––– Generating text predictor specs ––––––––")

        noteevents = self._get_tfidf_all_notes_specs(
            resolve_multiple="concatenate",
            interval_days=2,
        )

        return [noteevents]

    def get_feature_specs(self) -> list[Union[TextPredictorSpec, PredictorSpec]]:
        """Get a spec set."""

        if self.min_set_for_debug:
            return self._get_temporal_predictor_specs()

        return (
            self._get_temporal_predictor_specs()
            + self._get_text_predictor_specs()
            + self._get_static_predictor_specs()
        )
