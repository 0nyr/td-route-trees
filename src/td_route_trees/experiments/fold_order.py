"""P4.1 order-sensitivity experiment: is NDCPWLF composition association-free in doubles?

Compares, bitwise, three ways of computing the route ready-time function
delta_r from the same factors on real MAMUT ATFs:

1. ``checker``   — the checker's own association (alpha then theta, left fold);
                   bit-identical to ``check_td_solution`` by the frozen-fold tests.
2. ``leaf_left`` — precomposed leaves ``theta ∘ alpha``, left fold.
3. ``balanced``  — the same leaves, balanced-tree association (what a BBT computes).

Outcome drives the Stream-4 correctness criterion (see
reports/design/td-route-trees-conventions.md, "Association order and bit-identity").

Usage (from the repo root, venv active, benchmarks env var set):
    python -m td_route_trees.experiments.fold_order --routes 200 --output fold_order.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from td_route_trees import _core, load_instance, to_core

DEFAULT_PICKS: list[tuple[str, str, str, int]] = [
    # (problem_type, family, size_dir, max_instances)
    ("TDVRPTW", "Dabia2013", "n=25", 6),
    ("TDVRPTW", "Dabia2013", "n=100", 4),
    ("TDVRP", "Dabia2013", "n=25", 4),
    ("TDVRPTW", "Rifki2020", "n=10", 4),
    ("TDVRPTW", "Rifki2020", "n=30", 2),
    ("TDVRP", "Rifki2020", "n=10", 2),
    ("TDVRPTW", "Ari2018", "n=15", 4),
    ("TDVRPTW", "Ari2018", "n=30", 2),
    ("TDVRPTW", "Vu2020", "n=59", 4),
    ("TDVRP", "Vu2020", "n=59", 2),
]


def benchmarks_root() -> Path:
    import os

    for var in ("MAMUT_ROUTING_BENCHMARKS_ROOT", "MAMUT_ROUTING_ROOT"):
        value = os.environ.get(var)
        if value:
            root = Path(value)
            if var == "MAMUT_ROUTING_ROOT":
                root = root / "benchmarks"
            if root.is_dir():
                return root
    raise SystemExit("set MAMUT_ROUTING_BENCHMARKS_ROOT to the benchmarks tree")


def route_corpus(instance_path: Path, num_customers: int, rng: random.Random, count: int) -> list[list[int]]:
    routes: list[list[int]] = []
    bks_path = instance_path.with_name(instance_path.name.replace(".vrp.json", ".bks.Duration.json"))
    if bks_path.is_file():
        data = json.loads(bks_path.read_text())
        routes.extend([list(map(int, r)) for r in data["routes"]])
    customers = list(range(1, num_customers + 1))
    while len(routes) < count:
        length = rng.randint(1, min(num_customers, 60))
        routes.append(rng.sample(customers, length))
    return routes


def compare(instance_path: Path, routes_per_instance: int, seed: int) -> dict:
    loaded = load_instance(instance_path)
    inst = to_core(loaded)
    rng = random.Random(seed)
    corpus = route_corpus(instance_path, loaded.instance.num_customers, rng, routes_per_instance)

    stats = {
        "instance": str(instance_path),
        "routes": len(corpus),
        "feasible_checker": 0,
        "leaf_left_bitwise_identical": 0,
        "balanced_bitwise_identical": 0,
        "leaf_left_duration_equal": 0,
        "balanced_duration_equal": 0,
        "feasibility_mismatches": 0,
        "max_abs_duration_diff_leaf_left": 0.0,
        "max_abs_duration_diff_balanced": 0.0,
    }
    for route in corpus:
        ref_xs, ref_ys = _core.route_delta_checker(inst, route)
        ref_empty = len(ref_xs) == 0
        if not ref_empty:
            stats["feasible_checker"] += 1
            ref_dur, _ = _core.min_shifted_image(ref_xs, ref_ys)
        for label in ("leaf_left", "balanced"):
            fn = _core.route_delta_leaf_left if label == "leaf_left" else _core.route_delta_balanced
            xs, ys = fn(inst, route)
            empty = len(xs) == 0
            if empty != ref_empty:
                stats["feasibility_mismatches"] += 1
                continue
            if empty:
                stats[f"{label}_bitwise_identical"] += 1
                stats[f"{label}_duration_equal"] += 1
                continue
            if xs == ref_xs and ys == ref_ys:
                stats[f"{label}_bitwise_identical"] += 1
            dur, _ = _core.min_shifted_image(xs, ys)
            if dur == ref_dur:
                stats[f"{label}_duration_equal"] += 1
            else:
                key = f"max_abs_duration_diff_{label}"
                stats[key] = max(stats[key], abs(dur - ref_dur))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=int, default=200, help="routes per instance (incl. BKS routes)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("fold_order.json"))
    args = parser.parse_args()

    root = benchmarks_root()
    results: list[dict] = []
    for problem_type, family, size_dir, max_instances in DEFAULT_PICKS:
        directory = root / problem_type / family / size_dir
        if not directory.is_dir():
            continue
        for instance_path in sorted(directory.glob("*.vrp.json"))[:max_instances]:
            stats = compare(instance_path, args.routes, args.seed)
            stats["problem_type"] = problem_type
            stats["family"] = family
            stats["size"] = size_dir
            results.append(stats)
            print(
                f"{problem_type}/{family}/{size_dir}/{instance_path.stem}: "
                f"{stats['routes']} routes, "
                f"leaf_left identical {stats['leaf_left_bitwise_identical']}/{stats['routes']}, "
                f"balanced identical {stats['balanced_bitwise_identical']}/{stats['routes']}, "
                f"duration equal {stats['leaf_left_duration_equal']}/{stats['balanced_duration_equal']}, "
                f"feas mismatch {stats['feasibility_mismatches']}, "
                f"max dur diff {stats['max_abs_duration_diff_leaf_left']:.3e}/"
                f"{stats['max_abs_duration_diff_balanced']:.3e}"
            )

    total = {
        "routes": sum(r["routes"] for r in results),
        "leaf_left_bitwise_identical": sum(r["leaf_left_bitwise_identical"] for r in results),
        "balanced_bitwise_identical": sum(r["balanced_bitwise_identical"] for r in results),
        "leaf_left_duration_equal": sum(r["leaf_left_duration_equal"] for r in results),
        "balanced_duration_equal": sum(r["balanced_duration_equal"] for r in results),
        "feasibility_mismatches": sum(r["feasibility_mismatches"] for r in results),
        "max_abs_duration_diff_leaf_left": max((r["max_abs_duration_diff_leaf_left"] for r in results), default=0.0),
        "max_abs_duration_diff_balanced": max((r["max_abs_duration_diff_balanced"] for r in results), default=0.0),
    }
    print("\nTOTAL:", json.dumps(total, indent=2))
    args.output.write_text(json.dumps({"total": total, "per_instance": results}, indent=2))
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
