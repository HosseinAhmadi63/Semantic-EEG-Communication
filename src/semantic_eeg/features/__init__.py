"""Filter-bank spatial features and budget-matched PCA messages."""

from semantic_eeg.features.fbcsp import (
    ExactFBCSPTorch,
    FBCSPPCA,
    fit_fbcsp_pca,
    transform_exact_fbcsp,
)

__all__ = ["ExactFBCSPTorch", "FBCSPPCA", "fit_fbcsp_pca", "transform_exact_fbcsp"]
