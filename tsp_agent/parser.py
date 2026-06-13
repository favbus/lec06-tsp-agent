from __future__ import annotations

import csv
import io
import json
import re
from typing import Any

from .models import City


def default_course_example() -> list[City]:
    return [
        City("0", 0.0, 0.0),
        City("1", 10.0, 0.0),
        City("2", 15.0, 20.0),
        City("3", 20.0, 5.0),
    ]


def distance_matrix_course_example() -> list[list[float]]:
    return [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ]


def city_to_dict(city: City) -> dict[str, Any]:
    return {"name": city.name, "x": city.x, "y": city.y}


def city_from_dict(data: dict[str, Any]) -> City:
    return City(str(data["name"]), float(data["x"]), float(data["y"]))


def parse_cities(text: str) -> list[City]:
    content = text.strip()
    if not content:
        raise ValueError("请输入城市坐标，或在页面中选择随机生成。")

    try:
        data = json.loads(content)
        if isinstance(data, dict) and "cities" in data:
            data = data["cities"]
        if isinstance(data, list):
            cities = []
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    name = str(item.get("name", item.get("city", i)))
                    x = float(item["x"])
                    y = float(item["y"])
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    name = str(item[0]) if len(item) >= 3 else str(i)
                    x = float(item[-2])
                    y = float(item[-1])
                else:
                    raise ValueError
                cities.append(City(name, x, y))
            return validate_cities(cities)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass

    rows = list(csv.reader(io.StringIO(content)))
    if len(rows) >= 2:
        parsed: list[City] = []
        for i, row in enumerate(rows):
            clean = [part.strip() for part in row if part.strip()]
            if not clean:
                continue
            if i == 0 and any(token.lower() in {"name", "city", "x", "y"} for token in clean):
                continue
            if len(clean) >= 3:
                parsed.append(City(clean[0], float(clean[1]), float(clean[2])))
            elif len(clean) == 2:
                parsed.append(City(str(len(parsed)), float(clean[0]), float(clean[1])))
        if parsed:
            return validate_cities(parsed)

    pattern = re.compile(
        r"(?P<name>[\w\u4e00-\u9fff-]+)?\s*[\(:：,，]\s*"
        r"(?P<x>-?\d+(?:\.\d+)?)\s*[,，]\s*"
        r"(?P<y>-?\d+(?:\.\d+)?)\s*\)?"
    )
    matches = pattern.finditer(content)
    cities = []
    for i, match in enumerate(matches):
        name = match.group("name") or str(i)
        cities.append(City(name.strip(), float(match.group("x")), float(match.group("y"))))

    if cities:
        return validate_cities(cities)

    raise ValueError("无法识别城市格式。支持 JSON、CSV，或类似 A(10,20) 的坐标文本。")


def validate_cities(cities: list[City]) -> list[City]:
    if len(cities) < 3:
        raise ValueError("TSP 至少需要 3 个城市。")
    names = [city.name for city in cities]
    if len(set(names)) != len(names):
        raise ValueError("城市名称不能重复。")
    return cities


def cities_to_text(cities: list[City]) -> str:
    return "\n".join(f"{city.name},{city.x:.3f},{city.y:.3f}" for city in cities)
