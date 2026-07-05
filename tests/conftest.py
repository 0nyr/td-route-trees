import json
import os
import random
from pathlib import Path

import pytest


def benchmarks_root() -> Path | None:
    """MAMUT-routing benchmarks tree, from the standard lib env vars."""
    for var in ("MAMUT_ROUTING_BENCHMARKS_ROOT", "MAMUT_ROUTING_ROOT"):
        value = os.environ.get(var)
        if value:
            root = Path(value)
            if var == "MAMUT_ROUTING_ROOT":
                root = root / "benchmarks"
            if root.is_dir():
                return root
    return None


def require_benchmarks() -> None:
    if benchmarks_root() is None:
        pytest.skip(
            "MAMUT-routing benchmarks not found: set MAMUT_ROUTING_BENCHMARKS_ROOT "
            "(or MAMUT_ROUTING_ROOT) to a MAMUT-routing checkout on the td branch"
        )


def sample_instance_paths() -> list[Path]:
    """Small deterministic per-family sample for the correctness suite."""
    root = benchmarks_root()
    if root is None:
        return []
    picks: list[tuple[str, str, str, list[str]]] = [
        ("TDVRPTW", "Dabia2013", "n=25", ["C101", "R101", "RC201"]),
        ("TDVRP", "Dabia2013", "n=25", ["C101", "R201"]),
        ("TDVRPTW", "Rifki2020", "n=10", []),
        ("TDVRP", "Rifki2020", "n=10", []),
        ("TDVRPTW", "Ari2018", "n=15", []),
        ("TDVRPTW", "Vu2020", "n=59", []),
    ]
    paths: list[Path] = []
    for problem_type, family, size_dir, names in picks:
        directory = root / problem_type / family / size_dir
        if not directory.is_dir():
            continue
        if names:
            paths.extend(p for name in names if (p := directory / f"{name}.vrp.json").is_file())
        else:
            paths.extend(sorted(directory.glob("*.vrp.json"))[:2])
    return paths


def random_routes(num_customers: int, rng: random.Random, count: int) -> list[list[int]]:
    """Random customer sequences: many will be time-infeasible, on purpose."""
    routes: list[list[int]] = []
    customers = list(range(1, num_customers + 1))
    for _ in range(count):
        length = rng.randint(1, min(num_customers, 40))
        routes.append(rng.sample(customers, length))
    return routes


def bks_routes(instance_path: Path) -> list[list[int]]:
    """Routes of the sibling Duration BKS, when present (deadline-riding corpus)."""
    bks_path = instance_path.with_name(
        instance_path.name.replace(".vrp.json", ".bks.Duration.json")
    )
    if not bks_path.is_file():
        return []
    data = json.loads(bks_path.read_text())
    return [list(map(int, route)) for route in data["routes"]]
