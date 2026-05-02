"""Surrogate model implementations."""

from surroforge.models.base import ModelNotFittedError, SurrogateModel
from surroforge.models.ensemble import EnsembleRegressor
from surroforge.models.field_autoencoder import FieldAutoencoderMLP
from surroforge.models.mlp import MLPRegressor, torch_available
from surroforge.models.sklearn import SklearnRegressor

__all__ = [
    "EnsembleRegressor",
    "FieldAutoencoderMLP",
    "MLPRegressor",
    "ModelNotFittedError",
    "SklearnRegressor",
    "SurrogateModel",
    "torch_available",
]
