"""Move-evaluation throughput microbenchmark: naive recomposition vs structures.

Times random exchange-move evaluations (C++-side loop, no Python overhead) on
BKS-route solutions of real MAMUT instances, per structure and per max segment
length. Checksums guard against silent divergence between methods (they may
differ in the documented ulp class only).

Usage:
    python -m td_route_trees.experiments.bench_moves --moves 20000 --output bench.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from td_route_trees import _core, load_instance, to_core

METHODS = {0: "naive", 1: "tree", 2: "lca"}

DEFAULT_PICKS: list[tuple[str, str, str, str]] = [
    ("TDVRPTW", "Dabia2013", "n=25", "C101"),
    ("TDVRPTW", "Dabia2013", "n=100", "C101"),
    ("TDVRPTW", "Dabia2013", "n=100", "R211"),
    ("TDVRPTW", "Dabia2013", "n=100", "RC203"),
    ("TDVRP", "Dabia2013", "n=100", "R211"),
    ("TDVRPTW", "Rifki2020", "n=30", "Rifki-1"),
    ("TDVRPTW", "Ari2018", "n=30", "Ari-A1-pB-d70-w25"),
    ("TDVRPTW", "Vu2020", "n=60", None),
]


def benchmarks_root() -> Path:
    for var in ("MAMUT_ROUTING_BENCHMARKS_ROOT", "MAMUT_ROUTING_ROOT"):
        value = os.environ.get(var)
        if value:
            root = Path(value)
            if var == "MAMUT_ROUTING_ROOT":
                root = root / "benchmarks"
            if root.is_dir():
                return root
    raise SystemExit("set MAMUT_ROUTING_BENCHMARKS_ROOT to the benchmarks tree")


def bks_routes(instance_path: Path) -> list[list[int]]:
    bks_path = instance_path.with_name(instance_path.name.replace(".vrp.json", ".bks.Duration.json"))
    if not bks_path.is_file():
        return []
    return [list(map(int, r)) for r in json.loads(bks_path.read_text())["routes"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moves", type=int, default=20000)
    parser.add_argument("--max-segs", type=int, nargs="+", default=[3, 8, 16])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("bench_moves.json"))
    args = parser.parse_args()

    root = benchmarks_root()
    results = []
    for problem_type, family, size_dir, name in DEFAULT_PICKS:
        directory = root / problem_type / family / size_dir
        if not directory.is_dir():
            continue
        if name is None:
            candidates = sorted(directory.glob("*.vrp.json"))
            if not candidates:
                continue
            instance_path = candidates[0]
        else:
            instance_path = directory / f"{name}.vrp.json"
            if not instance_path.is_file():
                continue
        routes = bks_routes(instance_path)
        if len(routes) < 2:
            continue
        loaded = load_instance(instance_path)
        inst = to_core(loaded)
        mean_len = sum(len(r) for r in routes) / len(routes)
        max_len = max(len(r) for r in routes)
        for max_seg in args.max_segs:
            row = {
                "instance": f"{problem_type}/{family}/{size_dir}/{instance_path.stem}",
                "routes": len(routes),
                "mean_route_len": round(mean_len, 1),
                "max_route_len": max_len,
                "max_seg": max_seg,
            }
            for method, label in METHODS.items():
                r = _core.bench_exchange(inst, routes, args.moves, max_seg, args.seed, method)
                row[f"{label}_ns_per_eval"] = round(r["ns_per_eval"], 1)
                row[f"{label}_checksum"] = r["checksum"]
                row[f"{label}_feasible"] = r["feasible"]
                if method > 0:
                    row[f"{label}_build_ns_total"] = round(r["build_ns_total"], 1)
            row["speedup_tree"] = round(row["naive_ns_per_eval"] / row["tree_ns_per_eval"], 2)
            row["speedup_lca"] = round(row["naive_ns_per_eval"] / row["lca_ns_per_eval"], 2)
            results.append(row)
            print(
                f"{row['instance']} (m̄={row['mean_route_len']}, k≤{max_seg}): "
                f"naive {row['naive_ns_per_eval']/1e3:.1f}µs, tree {row['tree_ns_per_eval']/1e3:.1f}µs (×{row['speedup_tree']}), "
                f"lca {row['lca_ns_per_eval']/1e3:.1f}µs (×{row['speedup_lca']}) "
                f"(feas {row['naive_feasible']}/{row['tree_feasible']}/{row['lca_feasible']})"
            )

    args.output.write_text(json.dumps(results, indent=2))
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
