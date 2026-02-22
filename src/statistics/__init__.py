"""Statistical foundations package for institutional data systems."""

from .probability_models import generate_synthetic_flight_delays
from .hypothesis_testing import welch_t_test, mean_confidence_interval
from .monte_carlo import run_monte_carlo_simulations

__all__ = [
    "generate_synthetic_flight_delays",
    "welch_t_test",
    "mean_confidence_interval",
    "run_monte_carlo_simulations",
]
