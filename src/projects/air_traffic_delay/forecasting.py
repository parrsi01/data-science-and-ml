"""Delay forecasting for operational planning (Prophet with ARIMA fallback)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _daily_delay_series(flights_df: pd.DataFrame) -> pd.DataFrame:
    df = flights_df.copy()
    df["scheduled_time"] = pd.to_datetime(df["scheduled_time"])
    daily = (
        df.assign(date=df["scheduled_time"].dt.floor("D"))
        .groupby("date", as_index=False)["delay_minutes"]
        .mean()
        .rename(columns={"date": "ds", "delay_minutes": "y"})
        .sort_values("ds", kind="mergesort")
        .reset_index(drop=True)
    )
    return daily


def _forecast_with_prophet(daily: pd.DataFrame, horizon_days: int) -> tuple[pd.DataFrame, str]:
    try:
        from prophet import Prophet  # type: ignore[import-not-found]
    except Exception as exc:  # catches stdlib-shadow import issues too
        raise RuntimeError(f"Prophet unavailable: {exc}") from exc

    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
    model.fit(daily[["ds", "y"]])
    future = model.make_future_dataframe(periods=horizon_days, freq="D")
    forecast = model.predict(future)
    out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    out["method"] = "prophet"
    return out, "prophet"


def _forecast_with_arima(daily: pd.DataFrame, horizon_days: int) -> tuple[pd.DataFrame, str]:
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore[import-not-found]

    series = daily.set_index("ds")["y"].astype(float)
    model = ARIMA(series, order=(2, 1, 2))
    fitted = model.fit()
    fc = fitted.get_forecast(steps=horizon_days)
    pred_mean = fc.predicted_mean
    conf = fc.conf_int()
    forecast_index = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    hist = daily[["ds", "y"]].copy()
    hist["yhat"] = hist["y"]
    hist["yhat_lower"] = np.nan
    hist["yhat_upper"] = np.nan
    hist["method"] = "arima"

    future = pd.DataFrame(
        {
            "ds": forecast_index,
            "y": np.nan,
            "yhat": pred_mean.to_numpy(dtype=float),
            "yhat_lower": conf.iloc[:, 0].to_numpy(dtype=float),
            "yhat_upper": conf.iloc[:, 1].to_numpy(dtype=float),
            "method": "arima",
        }
    )
    out = pd.concat([hist, future], ignore_index=True)
    return out, "arima"


def run_delay_forecast(
    flights_df: pd.DataFrame,
    *,
    enabled: bool,
    horizon_days: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run delay forecasting and save CSV/PNG artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily = _daily_delay_series(flights_df)
    if not enabled:
        return {"enabled": False, "daily_rows": int(len(daily))}

    method = "arima"
    forecast_df: pd.DataFrame
    notes = ""
    try:
        forecast_df, method = _forecast_with_prophet(daily, horizon_days)
        notes = "Prophet forecast completed"
    except Exception as exc:
        forecast_df, method = _forecast_with_arima(daily, horizon_days)
        notes = f"Prophet unavailable; ARIMA fallback used ({exc})"

    csv_path = output_dir / "delay_forecast.csv"
    forecast_df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(daily["ds"], daily["y"], label="Observed Daily Mean Delay", color="#0B6E4F")
    future_mask = forecast_df["y"].isna() if "y" in forecast_df.columns else ~forecast_df["ds"].isin(daily["ds"])
    ax.plot(forecast_df["ds"], forecast_df["yhat"], label=f"Forecast ({method.upper()})", color="#C84C09")
    if {"yhat_lower", "yhat_upper"} <= set(forecast_df.columns):
        mask = forecast_df["yhat_lower"].notna() & forecast_df["yhat_upper"].notna()
        ax.fill_between(
            forecast_df.loc[mask, "ds"],
            forecast_df.loc[mask, "yhat_lower"],
            forecast_df.loc[mask, "yhat_upper"],
            color="#C84C09",
            alpha=0.18,
            label="Forecast Interval",
        )
    ax.set_title("Daily Mean Delay Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Delay Minutes")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    png_path = output_dir / "delay_forecast.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    future_rows = forecast_df[forecast_df["ds"] > daily["ds"].max()].copy()
    trend_delta = float(future_rows["yhat"].iloc[-1] - future_rows["yhat"].iloc[0]) if len(future_rows) >= 2 else 0.0
    trend_label = "rising" if trend_delta > 0.2 else "falling" if trend_delta < -0.2 else "stable"
    summary = {
        "enabled": True,
        "method": method,
        "notes": notes,
        "horizon_days": int(horizon_days),
        "daily_rows": int(len(daily)),
        "forecast_rows": int(len(forecast_df)),
        "trend_delta_minutes": trend_delta,
        "trend_label": trend_label,
        "artifacts": {"forecast_csv": str(csv_path), "forecast_png": str(png_path)},
    }
    meta_path = output_dir / "forecast_meta.json"
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["artifacts"]["forecast_meta_json"] = str(meta_path)
    return summary

