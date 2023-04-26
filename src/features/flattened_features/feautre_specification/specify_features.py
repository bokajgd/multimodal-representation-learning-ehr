"""Feature specification module."""
import logging
import sys

sys.path.append(".")

from psycop_feature_generation.application_modules.project_setup import ProjectInfo
from timeseriesflattener.feature_spec_objects import (
    BaseModel,
    OutcomeGroupSpec,
    OutcomeSpec,
    PredictorGroupSpec,
    PredictorSpec,
    StaticSpec,
    _AnySpec,
)

log = logging.getLogger(__name__)


class SpecSet(BaseModel):
    """A set of unresolved specs, ready for resolving."""

    temporal_predictors: list[PredictorSpec]
    metadata: list[_AnySpec]


class FeatureSpecifier:
    def __init__(self, project_info: ProjectInfo):
        self.project_info = project_info

    def _get_admissions_specs(
        self,
        resolve_multiple=None,
        interval_days=None,
    ):
        """Get admissions specs."""

        if resolve_multiple is None:
            resolve_multiple = ["count", "mean"]

        if interval_days is None:
            interval_days = [182, 365]

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

        if resolve_multiple is None:
            resolve_multiple = ["max", "min", "mean"]

        if interval_days is None:
            interval_days = [0.146, 2, 30]

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

        if resolve_multiple is None:
            resolve_multiple = ["latest", "mean", "change_per_day"]

        if interval_days is None:
            interval_days = [1, 2]

        log.info("–––––––– Generating chartevents specs ––––––––")

        admissions = PredictorGroupSpec(
            values_loader=("gcs",),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[0],
            allowed_nan_value_prop=[0],
        ).create_combinations()

        return admissions

    def _get_temporal_predictor_specs(self) -> list[PredictorSpec]:
        """Generate predictor spec list."""
        log.info("–––––––– Generating temporal predictor specs ––––––––")

        admissions = self._get_admissions_specs()

        inputevents = self._get_inputevents_specs()

        chartevents = self._get_chartevents_specs()

        return admissions + inputevents + chartevents

    def get_feature_specs(self) -> list[_AnySpec]:
        """Get a spec set."""

        return self._get_temporal_predictor_specs()
