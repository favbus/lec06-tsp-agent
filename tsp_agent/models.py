from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict


@dataclass(frozen=True)
class City:
    name: str
    x: float
    y: float


@dataclass(frozen=True)
class SAParams:
    initial_temperature: float = 1000.0
    cooling_rate: float = 0.92
    chain_length: int = 500
    final_temperature: float = 0.01
    max_temperature_steps: int = 300
    seed: int = 42


class AgentState(TypedDict, total=False):
    user_request: str
    cities: list[dict[str, Any]]
    distance_matrix: list[list[float]]
    params: dict[str, Any]
    plan: str
    solution: dict[str, Any]
    summary: str
    report_path: str
    chart_path: str
    route_chart_path: str
    logs: list[str]
