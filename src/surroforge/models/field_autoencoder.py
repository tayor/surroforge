"""Field surrogate placeholder with a stable public API."""

from __future__ import annotations


class FieldAutoencoderMLP:
    """Reserved API for compressed field-output surrogates.

    Field surrogates need mesh/field layout decisions that are intentionally kept
    separate from the scalar-output v0.1 path. The class exists so user code can
    feature-detect the planned API and receive a clear message.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "FieldAutoencoderMLP is planned for the field-output workflow. "
            "Use MLPRegressor or EnsembleRegressor for scalar/vector outputs in v0.1."
        )
