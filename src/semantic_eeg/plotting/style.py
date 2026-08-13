"""Shared visual identity for publication figures."""

from __future__ import annotations

import matplotlib as mpl

METHOD_SEMANTIC = "Semantic residual"
METHOD_FBCSP = "FBCSP + PCA"
METHOD_RECONSTRUCTION = "Reconstruction latent"
METHOD_RECEIVER_ONLY = "Receiver-only"

COLORS = {
    METHOD_SEMANTIC: "#0072B2",
    METHOD_FBCSP: "#D55E00",
    METHOD_RECONSTRUCTION: "#7A7A7A",
    METHOD_RECEIVER_ONLY: "#009E73",
}

MARKERS = {
    METHOD_SEMANTIC: "o",
    METHOD_FBCSP: "s",
    METHOD_RECONSTRUCTION: "^",
    METHOD_RECEIVER_ONLY: "D",
}


def apply_publication_style() -> None:
    """Apply the shared publication typography, colors, and axis styling."""

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11.5,
            "axes.titlesize": 13.0,
            "axes.labelsize": 12.0,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.8,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )
