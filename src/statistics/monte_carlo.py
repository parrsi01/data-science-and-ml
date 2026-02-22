"""Monte Carlo simulations for institutional aviation analytics use cases."""

from __future__ import annotations

from pathlib import Path
import math
import random
import struct
import zlib
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = PROJECT_ROOT / "reports" / "monte_carlo_simulation.png"


def _write_simple_png(
    path: Path,
    width: int,
    height: int,
    rgb_pixels: list[tuple[int, int, int]],
) -> None:
    """Write a minimal RGB PNG file using only stdlib modules."""

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
        raw.append(0)
        for r, g, b in rgb_pixels[y * width : (y + 1) * width]:
            raw.extend((r, g, b))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    payload = bytearray(b"\x89PNG\r\n\x1a\n")
    payload.extend(chunk(b"IHDR", ihdr))
    payload.extend(chunk(b"IDAT", zlib.compress(bytes(raw), level=9)))
    payload.extend(chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(payload))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("At least one value is required")
    return sum(values) / len(values)


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("At least two values are required")
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def _poisson_knuth(rate_lambda: float, rng: random.Random) -> int:
    """Sample from Poisson(lambda) using Knuth's algorithm."""

    if rate_lambda <= 0:
        raise ValueError("rate_lambda must be positive")
    threshold = math.exp(-rate_lambda)
    product = 1.0
    k = 0
    while product > threshold:
        k += 1
        product *= rng.random()
    return k - 1


def simulate_airline_congestion_probability(
    iterations: int = 10_000,
    mean_flights_per_hour: float = 58.0,
    hourly_capacity: int = 60,
    seed: int = 42,
) -> dict[str, object]:
    """Estimate congestion probability (demand > capacity) via Monte Carlo."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    rng = random.Random(seed)
    demand_samples: list[int] = []
    congestion_flags: list[int] = []

    for _ in range(iterations):
        demand = _poisson_knuth(mean_flights_per_hour, rng)
        demand_samples.append(demand)
        congestion_flags.append(1 if demand > hourly_capacity else 0)

    congestion_probability = sum(congestion_flags) / iterations
    return {
        "iterations": iterations,
        "hourly_capacity": hourly_capacity,
        "demand_samples": demand_samples,
        "congestion_probability": congestion_probability,
        "mean_demand": _mean([float(x) for x in demand_samples]),
    }


def simulate_fuel_cost_variation(
    iterations: int = 10_000,
    flight_fuel_burn_liters: float = 5200.0,
    fuel_price_mean: float = 0.84,
    fuel_price_std: float = 0.07,
    seed: int = 43,
) -> dict[str, object]:
    """Simulate fuel cost variation using random fuel prices."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if fuel_price_std <= 0:
        raise ValueError("fuel_price_std must be positive")
    rng = random.Random(seed)
    price_samples: list[float] = []
    cost_samples: list[float] = []

    for _ in range(iterations):
        price = max(0.0, rng.gauss(fuel_price_mean, fuel_price_std))
        price_samples.append(price)
        cost_samples.append(flight_fuel_burn_liters * price)

    return {
        "iterations": iterations,
        "price_samples": price_samples,
        "cost_samples": cost_samples,
        "mean_cost": _mean(cost_samples),
        "std_cost": _sample_std(cost_samples),
        "min_cost": min(cost_samples),
        "max_cost": max(cost_samples),
    }


def _draw_histogram_panel(
    pixels: list[tuple[int, int, int]],
    width: int,
    height: int,
    panel_left: int,
    panel_top: int,
    panel_width: int,
    panel_height: int,
    values: Sequence[float],
    bar_color: tuple[int, int, int],
) -> None:
    """Draw a simple histogram panel onto a pixel buffer."""

    bg = (255, 255, 255)
    axis = (148, 163, 184)
    for y in range(panel_top, panel_top + panel_height):
        for x in range(panel_left, panel_left + panel_width):
            pixels[y * width + x] = bg

    x_axis_y = panel_top + panel_height - 25
    y_axis_x = panel_left + 30
    for x in range(y_axis_x, panel_left + panel_width - 10):
        pixels[x_axis_y * width + x] = axis
    for y in range(panel_top + 10, x_axis_y + 1):
        pixels[y * width + y_axis_x] = axis

    if not values:
        return

    bins = 24
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        vmax = vmin + 1.0
    bin_width = (vmax - vmin) / bins
    counts = [0] * bins
    for value in values:
        idx = int((value - vmin) / bin_width)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1

    max_count = max(counts) or 1
    plot_left = y_axis_x + 5
    plot_right = panel_left + panel_width - 12
    plot_top = panel_top + 12
    plot_bottom = x_axis_y - 4
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    bar_px_width = max(1, plot_width // bins)

    for idx, count in enumerate(counts):
        bar_height = int((count / max_count) * plot_height)
        x0 = plot_left + idx * bar_px_width
        x1 = min(plot_right, x0 + max(1, bar_px_width - 2))
        y0 = plot_bottom - bar_height
        for y in range(y0, plot_bottom):
            for x in range(x0, x1):
                pixels[y * width + x] = bar_color


def save_monte_carlo_histogram(
    congestion_samples: Sequence[int],
    fuel_cost_samples: Sequence[float],
    output_path: Path | None = None,
) -> Path:
    """Save a two-panel histogram PNG for Monte Carlo simulation outcomes."""

    target = output_path or REPORT_PATH
    width, height = 900, 500
    canvas = [(244, 247, 250) for _ in range(width * height)]

    _draw_histogram_panel(
        canvas,
        width,
        height,
        panel_left=20,
        panel_top=20,
        panel_width=410,
        panel_height=460,
        values=[float(v) for v in congestion_samples],
        bar_color=(37, 99, 235),
    )
    _draw_histogram_panel(
        canvas,
        width,
        height,
        panel_left=470,
        panel_top=20,
        panel_width=410,
        panel_height=460,
        values=list(fuel_cost_samples),
        bar_color=(5, 150, 105),
    )

    _write_simple_png(target, width, height, canvas)
    return target


def run_monte_carlo_simulations(iterations: int = 10_000) -> dict[str, object]:
    """Run Monte Carlo simulations and save histogram figure."""

    congestion = simulate_airline_congestion_probability(iterations=iterations)
    fuel = simulate_fuel_cost_variation(iterations=iterations)
    figure_path = save_monte_carlo_histogram(
        congestion_samples=congestion["demand_samples"],  # type: ignore[arg-type]
        fuel_cost_samples=fuel["cost_samples"],  # type: ignore[arg-type]
        output_path=REPORT_PATH,
    )

    return {
        "iterations": iterations,
        "congestion": congestion,
        "fuel_cost": fuel,
        "figure_path": str(figure_path),
    }
