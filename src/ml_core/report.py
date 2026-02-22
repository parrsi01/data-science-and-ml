"""CLI helper for printing ML top-line metrics."""

from __future__ import annotations

from pathlib import Path
import json


def main() -> None:
    metrics_path = Path("reports/ml_core/metrics.json")
    if not metrics_path.exists():
        raise SystemExit("metrics.json not found; run `make ml-train` first")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for model_name, payload in metrics.items():
        vals = payload["test_metrics"]
        print(
            f"{model_name}: "
            f"accuracy={vals['accuracy']:.4f} "
            f"precision={vals['precision']:.4f} "
            f"recall={vals['recall']:.4f} "
            f"f1={vals['f1']:.4f} "
            f"roc_auc={vals['roc_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
