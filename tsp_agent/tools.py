from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from .models import City, SAParams
from .sa import simulated_annealing_tsp


def _coerce_params(data: dict[str, Any]) -> SAParams:
    return SAParams(
        initial_temperature=float(data.get("initial_temperature", 1000.0)),
        cooling_rate=float(data.get("cooling_rate", 0.92)),
        chain_length=int(data.get("chain_length", 500)),
        final_temperature=float(data.get("final_temperature", 0.01)),
        max_temperature_steps=int(data.get("max_temperature_steps", 300)),
        seed=int(data.get("seed", 42)),
    )


def _coerce_cities(data: list[dict[str, Any]]) -> list[City]:
    return [City(str(item["name"]), float(item["x"]), float(item["y"])) for item in data]


@tool("solve_tsp_by_simulated_annealing")
def solve_tsp_by_simulated_annealing(
    cities: list[dict[str, Any]],
    params: dict[str, Any],
    distance_matrix: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Solve a TSP instance with simulated annealing and return route, metrics, and iteration history."""
    return simulated_annealing_tsp(_coerce_cities(cities), _coerce_params(params), distance_matrix)
