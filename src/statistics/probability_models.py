"""Probability models and synthetic data generation for institutional analytics.

This module implements normal, binomial, and Poisson utilities using the Python
standard library so it remains functional in restricted offline environments.
"""

from __future__ import annotations

from pathlib import Path
import math
import random
import zlib
import struct
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
NORMAL_VISUALIZATION_PATH = REPORTS_DIR / "normal_distribution.png"


def _mean(values: Iterable[float]) -> float:
    """Compute the arithmetic mean."""

    data = list(values)
    if not data:
        raise ValueError("Mean requires at least one value")
    return sum(data) / len(data)


def _sample_std(values: Iterable[float]) -> float:
    """Compute sample standard deviation (n-1 denominator)."""

    data = list(values)
    if len(data) < 2:
        raise ValueError("Sample standard deviation requires at least two values")
    mu = _mean(data)
    variance = sum((x - mu) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)


def _write_simple_png(
    path: Path,
    width: int,
    height: int,
    rgb_pixels: list[tuple[int, int, int]],
) -> None:
    """Write a minimal RGB PNG file using only the standard library."""

    if len(rgb_pixels) != width * height:
        raise ValueError("Pixel count does not match image dimensions")

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    raw = bytearray()
    for y in range(height):
        raw.append(0)  # No filter byte per PNG scanline.
        row = rgb_pixels[y * width : (y + 1) * width]
        for r, g, b in row:
            raw.extend((r, g, b))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(chunk(b"IHDR", ihdr))
    png.extend(chunk(b"IDAT", zlib.compress(bytes(raw), level=9)))
    png.extend(chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(png))


def normal_pdf(x: float, mean: float = 0.0, std_dev: float = 1.0) -> float:
    """Return the Normal distribution PDF at ``x``."""

    if std_dev <= 0:
        raise ValueError("std_dev must be positive")
    z = (x - mean) / std_dev
    coefficient = 1.0 / (std_dev * math.sqrt(2.0 * math.pi))
    return coefficient * math.exp(-0.5 * z * z)


def normal_cdf(x: float, mean: float = 0.0, std_dev: float = 1.0) -> float:
    """Return the Normal distribution CDF at ``x`` using ``math.erf``."""

    if std_dev <= 0:
        raise ValueError("std_dev must be positive")
    z = (x - mean) / (std_dev * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def normal_distribution_points(
    mean: float = 0.0,
    std_dev: float = 1.0,
    x_min: float = -4.0,
    x_max: float = 4.0,
    step: float = 0.05,
) -> list[tuple[float, float]]:
    """Generate (x, pdf) points for a Normal curve."""

    if step <= 0:
        raise ValueError("step must be positive")
    points: list[tuple[float, float]] = []
    x = x_min
    while x <= x_max + 1e-12:
        points.append((x, normal_pdf(x, mean=mean, std_dev=std_dev)))
        x += step
    return points


def save_normal_distribution_visualization(
    path: Path | None = None,
    mean: float = 0.0,
    std_dev: float = 1.0,
) -> Path:
    """Save a simple PNG visualization of the Normal PDF curve.

    The renderer is intentionally minimal to keep the project runnable without
    Matplotlib in offline environments.
    """

    target = path or NORMAL_VISUALIZATION_PATH
    width, height = 640, 360
    bg = (248, 250, 252)
    axis = (148, 163, 184)
    curve = (30, 64, 175)

    pixels = [bg for _ in range(width * height)]

    # Draw axes.
    x_axis_y = height - 40
    y_axis_x = 50
    for x in range(y_axis_x, width - 20):
        pixels[x_axis_y * width + x] = axis
    for y in range(20, x_axis_y + 1):
        pixels[y * width + y_axis_x] = axis

    points = normal_distribution_points(mean=mean, std_dev=std_dev, x_min=-4, x_max=4, step=0.01)
    max_pdf = max(p[1] for p in points)

    prev_px: tuple[int, int] | None = None
    for x, y in points:
        px = y_axis_x + int(((x + 4.0) / 8.0) * (width - y_axis_x - 30))
        py = x_axis_y - int((y / max_pdf) * (x_axis_y - 30))
        if 0 <= px < width and 0 <= py < height:
            pixels[py * width + px] = curve
            if prev_px is not None:
                x0, y0 = prev_px
                dx = abs(px - x0)
                dy = abs(py - y0)
                steps = max(dx, dy, 1)
                for i in range(steps + 1):
                    ix = x0 + (px - x0) * i // steps
                    iy = y0 + (py - y0) * i // steps
                    if 0 <= ix < width and 0 <= iy < height:
                        pixels[iy * width + ix] = curve
            prev_px = (px, py)

    _write_simple_png(target, width, height, pixels)
    return target


def binomial_pmf(k: int, n: int, p: float) -> float:
    """Return the Binomial PMF P(X=k) for ``n`` trials with success ``p``."""

    if not (0 <= k <= n):
        return 0.0
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be between 0 and 1")
    return math.comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))


def airline_on_time_binomial_example(
    n_flights: int = 20,
    p_on_time: float = 0.82,
    threshold: int = 16,
) -> dict[str, float | int]:
    """Example Binomial calculation for airline on-time performance.

    Returns probabilities for exactly ``threshold`` on-time flights and for
    meeting/exceeding the threshold.
    """

    exact = binomial_pmf(threshold, n_flights, p_on_time)
    at_least = sum(binomial_pmf(k, n_flights, p_on_time) for k in range(threshold, n_flights + 1))
    return {
        "n_flights": n_flights,
        "p_on_time": p_on_time,
        "threshold": threshold,
        "p_exactly_threshold": exact,
        "p_at_least_threshold": at_least,
    }


def poisson_pmf(k: int, rate_lambda: float) -> float:
    """Return Poisson PMF P(X=k) for rate ``rate_lambda``."""

    if k < 0:
        return 0.0
    if rate_lambda <= 0:
        raise ValueError("rate_lambda must be positive")
    return math.exp(-rate_lambda) * (rate_lambda**k) / math.factorial(k)


def cern_style_rare_event_simulation(
    intervals: int = 1000,
    rate_lambda: float = 0.7,
    seed: int = 42,
) -> dict[str, float | int]:
    """Simulate rare event counts per interval using Knuth's Poisson sampler."""

    if intervals <= 0:
        raise ValueError("intervals must be positive")
    rng = random.Random(seed)
    counts: list[int] = []
    for _ in range(intervals):
        # Knuth algorithm: counts independent events in a fixed interval.
        limit = math.exp(-rate_lambda)
        product = 1.0
        k = 0
        while product > limit:
            k += 1
            product *= rng.random()
        counts.append(k - 1)

    return {
        "intervals": intervals,
        "rate_lambda": rate_lambda,
        "mean_count": _mean(float(c) for c in counts),
        "max_count": max(counts),
    }


def generate_synthetic_flight_delays(
    count: int = 10_000,
    mean_delay: float = 12.0,
    std_dev: float = 8.0,
    seed: int = 42,
    clip_non_negative: bool = False,
) -> list[float]:
    """Generate synthetic flight delays with approximately controlled mean/std.

    The raw Gaussian sample is linearly rescaled so the final sample matches the
    requested mean and sample standard deviation closely.
    """

    if count < 2:
        raise ValueError("count must be at least 2")
    if std_dev <= 0:
        raise ValueError("std_dev must be positive")

    rng = random.Random(seed)
    raw = [rng.gauss(mean_delay, std_dev) for _ in range(count)]
    raw_mean = _mean(raw)
    raw_std = _sample_std(raw)
    scaled = [((x - raw_mean) / raw_std) * std_dev + mean_delay for x in raw]

    # Optional clipping is useful for some operational use cases, but clipping
    # changes the achieved mean/std, so it is disabled by default.
    if clip_non_negative:
        return [max(0.0, round(x, 3)) for x in scaled]
    return [round(x, 3) for x in scaled]


def summarize_synthetic_delays(delays: list[float]) -> dict[str, float | int]:
    """Return a compact summary of synthetic delay values."""

    return {
        "count": len(delays),
        "mean_delay": round(_mean(delays), 4),
        "std_dev": round(_sample_std(delays), 4),
        "min_delay": round(min(delays), 4),
        "max_delay": round(max(delays), 4),
    }
