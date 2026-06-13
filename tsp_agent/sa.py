from __future__ import annotations

import math
import random
from typing import Any

from .models import City, SAParams


def euclidean_distance(a: City, b: City) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def build_distance_matrix(cities: list[City]) -> list[list[float]]:
    return [[euclidean_distance(a, b) for b in cities] for a in cities]


def route_length(route: list[int], distance_matrix: list[list[float]]) -> float:
    total = 0.0
    n = len(route)
    for i in range(n):
        total += distance_matrix[route[i]][route[(i + 1) % n]]
    return total


def nearest_neighbor_route(distance_matrix: list[list[float]], start: int = 0) -> list[int]:
    n = len(distance_matrix)
    unvisited = set(range(n))
    route = [start]
    unvisited.remove(start)
    while unvisited:
        current = route[-1]
        next_city = min(unvisited, key=lambda idx: distance_matrix[current][idx])
        route.append(next_city)
        unvisited.remove(next_city)
    return route


def two_swap(route: list[int], rng: random.Random) -> tuple[list[int], tuple[int, int]]:
    new_route = route.copy()
    i, j = sorted(rng.sample(range(1, len(route)), 2))
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route, (i, j)


def normalize_route(route: list[int], start: int = 0) -> list[int]:
    if start not in route:
        return route
    index = route.index(start)
    return route[index:] + route[:index]


def simulated_annealing_tsp(
    cities: list[City],
    params: SAParams,
    distance_matrix: list[list[float]] | None = None,
) -> dict[str, Any]:
    if len(cities) < 3:
        raise ValueError("TSP 至少需要 3 个城市。")

    matrix = distance_matrix or build_distance_matrix(cities)
    rng = random.Random(params.seed)
    current_route = nearest_neighbor_route(matrix, start=0)
    tail = current_route[1:]
    rng.shuffle(tail)
    current_route = current_route[:1] + tail

    current_length = route_length(current_route, matrix)
    initial_length = current_length
    best_route = current_route.copy()
    best_length = current_length

    temperature = params.initial_temperature
    history = []
    sample_steps = []
    total_iterations = 0
    accepted = 0
    worse_accepted = 0
    improved = 0
    temperature_step = 0

    while (
        temperature > params.final_temperature
        and temperature_step < params.max_temperature_steps
    ):
        accepted_at_temp = 0
        best_before = best_length
        for inner in range(params.chain_length):
            candidate_route, swapped = two_swap(current_route, rng)
            candidate_length = route_length(candidate_route, matrix)
            delta = candidate_length - current_length
            probability = 1.0 if delta < 0 else math.exp(-delta / temperature)
            draw = rng.random()
            should_accept = delta < 0 or draw < probability

            if should_accept:
                current_route = candidate_route
                current_length = candidate_length
                accepted += 1
                accepted_at_temp += 1
                if delta >= 0:
                    worse_accepted += 1

            if current_length < best_length:
                best_route = current_route.copy()
                best_length = current_length
                improved += 1

            total_iterations += 1
            if len(sample_steps) < 12 and (inner < 3 or delta < 0):
                sample_steps.append(
                    {
                        "temperature_step": temperature_step + 1,
                        "inner_iteration": inner + 1,
                        "temperature": temperature,
                        "swapped_positions": swapped,
                        "candidate_length": candidate_length,
                        "current_length_after": current_length,
                        "delta": delta,
                        "acceptance_probability": probability,
                        "random_draw": draw,
                        "accepted": should_accept,
                        "best_length_after": best_length,
                    }
                )

        history.append(
            {
                "temperature_step": temperature_step + 1,
                "temperature": temperature,
                "current_length": current_length,
                "best_length": best_length,
                "accepted": accepted_at_temp,
                "acceptance_rate": accepted_at_temp / params.chain_length,
                "improved_this_step": best_length < best_before,
            }
        )
        temperature *= params.cooling_rate
        temperature_step += 1

    best_route = normalize_route(best_route, start=0)
    route_names = [cities[index].name for index in best_route]
    closed_route_names = route_names + [route_names[0]]
    improvement_rate = (
        (initial_length - best_length) / initial_length
        if initial_length > 0
        else 0.0
    )

    return {
        "city_count": len(cities),
        "cities": [{"name": c.name, "x": c.x, "y": c.y} for c in cities],
        "params": {
            "initial_temperature": params.initial_temperature,
            "cooling_rate": params.cooling_rate,
            "chain_length": params.chain_length,
            "final_temperature": params.final_temperature,
            "max_temperature_steps": params.max_temperature_steps,
            "seed": params.seed,
        },
        "distance_mode": "distance_matrix" if distance_matrix else "euclidean_coordinates",
        "distance_matrix": matrix if distance_matrix and len(matrix) <= 12 else None,
        "best_route": best_route,
        "best_route_names": route_names,
        "closed_route_names": closed_route_names,
        "best_length": best_length,
        "initial_length": initial_length,
        "improvement_rate": improvement_rate,
        "history": history,
        "sample_steps": sample_steps,
        "total_iterations": total_iterations,
        "accepted": accepted,
        "worse_accepted": worse_accepted,
        "improved": improved,
    }
