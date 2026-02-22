"""SHAP explainability artifact generation for tuned XGBoost models."""

from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import shap


def generate_shap_reports(model, X_sample, feature_names, output_dir: str | Path) -> dict[str, str]:
    """Generate SHAP summary artifacts and plain-language notes.

    Args:
        model: Trained tree model (XGBoost classifier).
        X_sample: Transformed feature matrix (dense or sparse).
        feature_names: Feature names aligned to transformed columns.
        output_dir: Output directory for SHAP artifacts.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(X_sample, "toarray"):
        X_dense = X_sample.toarray()
    else:
        X_dense = np.asarray(X_sample)

    feature_names = list(feature_names)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_dense)

    values = np.asarray(shap_values)
    if values.ndim == 3:
        # SHAP may return [classes, rows, cols] for some binary classifier setups.
        values = values[-1]

    summary_png = out_dir / "shap_summary.png"
    bar_png = out_dir / "shap_bar.png"
    values_npz = out_dir / "shap_values.npz"
    notes_md = out_dir / "explainability_notes.md"

    plt.figure(figsize=(8, 5))
    shap.summary_plot(values, X_dense, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(summary_png, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    shap.summary_plot(values, X_dense, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(bar_png, dpi=150, bbox_inches="tight")
    plt.close()

    np.savez_compressed(values_npz, shap_values=values)

    notes_md.write_text(
        "\n".join(
            [
                "# Explainability Notes (SHAP)",
                "",
                "- SHAP values estimate how each feature pushes predictions toward or away from the rare-event class.",
                "- `shap_summary.png` shows per-sample effect spread for each feature.",
                "- `shap_bar.png` shows average absolute impact (global importance).",
                "- Interpret high-impact features with domain context; importance alone does not prove causality.",
                "- Review one-hot encoded categorical features together when summarizing institutional decisions.",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "shap_summary_png": str(summary_png),
        "shap_bar_png": str(bar_png),
        "shap_values_npz": str(values_npz),
        "explainability_notes_md": str(notes_md),
    }
