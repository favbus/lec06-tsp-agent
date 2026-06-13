from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .models import City


def save_convergence_chart(solution: dict[str, Any], output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history = solution["history"]
    steps = [item["temperature_step"] for item in history]
    best = [item["best_length"] for item in history]
    current = [item["current_length"] for item in history]

    plt.figure(figsize=(8, 4.5), dpi=160)
    plt.plot(steps, current, color="#7c8aa0", linewidth=1.4, label="Current route")
    plt.plot(steps, best, color="#0f766e", linewidth=2.2, label="Best route")
    plt.xlabel("Temperature step")
    plt.ylabel("Route length")
    plt.title("Simulated Annealing Convergence")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return str(path)


def save_route_chart(cities: list[City], solution: dict[str, Any], output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    route = solution["best_route"]
    closed = route + [route[0]]
    xs = [cities[index].x for index in closed]
    ys = [cities[index].y for index in closed]

    plt.figure(figsize=(7, 5.2), dpi=160)
    plt.plot(xs, ys, color="#2563eb", linewidth=1.8)
    plt.scatter([c.x for c in cities], [c.y for c in cities], s=48, color="#dc2626", zorder=3)
    for index, city in enumerate(cities):
        plt.annotate(city.name, (city.x, city.y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Best TSP Route")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return str(path)
