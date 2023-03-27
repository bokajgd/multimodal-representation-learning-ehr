"""Feature specification module."""
import logging
import sys
sys.path.append(".")

from psycop_feature_generation.application_modules.project_setup import ProjectInfo

from timeseriesflattener.feature_spec_objects import (BaseModel,
                                                      OutcomeGroupSpec,
                                                      OutcomeSpec,
                                                      PredictorGroupSpec,
                                                      PredictorSpec,
                                                      StaticSpec, _AnySpec)


log = logging.getLogger(__name__)

class SpecSet(BaseModel):
    """A set of unresolved specs, ready for resolving."""

    temporal_predictors: list[PredictorSpec]
    metadata: list[_AnySpec]


class FeatureSpecifier:
    def __init__(self, project_info: ProjectInfo,):
        self.project_info = project_info

    def _get_admissions_specs(
        self, resolve_multiple, interval_days, allowed_nan_value_prop
    ):
        """Get admissions specs."""
        log.info("–––––––– Generating admissions specs ––––––––")

        admissions = PredictorGroupSpec(
            values_loader=("emergency_admissions",),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
        ).create_combinations()

        return admissions

    def _get_temporal_predictor_specs(self) -> list[PredictorSpec]:
        """Generate predictor spec list."""
        log.info("–––––––– Generating temporal predictor specs ––––––––")

        resolve_multiple = ["max", "min", "mean"]
        interval_days = [30, 180, 730]
        allowed_nan_value_prop = [0]

        admissions = self._get_admissions_specs(
            resolve_multiple,
            interval_days,
            allowed_nan_value_prop,
        )

        return admissions


    def get_feature_specs(self) -> list[_AnySpec]:
        """Get a spec set."""

        return (
            self._get_temporal_predictor_specs()
        )