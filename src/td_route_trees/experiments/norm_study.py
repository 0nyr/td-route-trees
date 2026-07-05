"""Norm-vs-no-norm checker study (2026-07-05/06 decision data).

Question (Onyr): should the MAMUT checker evolve to normalize NDCPWLFs after
composition, as paper0's visser+nor does? Four engine variants are measured
on real MAMUT routes (see cpp/structures/norm.h for the exact predicates):

    0 none       — checker status quo
    1 safe       — exact flat/vertical interior dedup (bitwise-neutral, gated)
    2 collinear  — exact chord-collinear pruning (ulp drift downstream)
    3 eps_slope  — pwlf_compare normalized_add predicate, EPS = 1e-6 (lossy)

Axes: breakpoint counts (final delta + max prefix during the fold), fold
timing, move-eval timing (naive / BBT / LCA), structure memory, Delta* drift
and feasibility flips vs V0, and long-chain growth (the "large-scale" probe:
sequences far longer than any feasible route).

Usage:
    python -m td_route_trees.experiments.norm_study --output norm_study.json \
        [--routes 200] [--moves 20000] [--fold-repeats 50] [--quick]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from td_route_trees import _core, load_instance, to_core
from td_route_trees.experiments.fold_order import DEFAULT_PICKS, benchmarks_root, route_corpus

MODES = {0: "none", 1: "safe", 2: "collinear", 3: "eps_slope", 4: "safe_int"}

# Long-route / heavy-bp instances the sorted-glob DEFAULT_PICKS never reach.
EXTRA_ACCURACY: list[tuple[str, str, str, str]] = [
    ("TDVRPTW", "Dabia2013", "n=100", "R211"),
    ("TDVRPTW", "Dabia2013", "n=100", "RC203"),
    ("TDVRPTW", "Dabia2013", "n=100", "R112"),
    ("TDVRP", "Dabia2013", "n=100", "R211"),
    ("TDVRP", "Rifki2020", "n=30", "Rifki-10"),
    ("TDVRP", "Ari2018", "n=30", "Ari-A10-pB-d70-w50"),
    ("TDVRPTW", "Vu2020", "n=99", None),
    ("TDVRP", "Vu2020", "n=99", None),
]

# Move-eval timing axis: the session-6 bench heroes + heavy-bp families.
BENCH_PICKS: list[tuple[str, str, str, str]] = [
    ("TDVRPTW", "Dabia2013", "n=100", "R211"),
    ("TDVRPTW", "Dabia2013", "n=100", "RC203"),
    ("TDVRP", "Dabia2013", "n=100", "R211"),
    ("TDVRPTW", "Rifki2020", "n=30", "Rifki-10"),
    ("TDVRPTW", "Ari2018", "n=30", "Ari-A10-pB-d70-w50"),
]

# Long-chain growth probe: TDVRP (no TW deadlines) so the only feasibility
# cap is the instance horizon itself — chains die when the accumulated ready
# time leaves the horizon, which *is* the structural growth bound.
CHAIN_PICKS: list[tuple[str, str, str, str | None]] = [
    ("TDVRP", "Dabia2013", "n=100", "R211"),
    ("TDVRP", "Rifki2020", "n=30", "Rifki-10"),
    ("TDVRP", "Ari2018", "n=30", "Ari-A10-pB-d70-w50"),
    ("TDVRP", "Vu2020", "n=59", None),  # first instance in the dir
]
CHAIN_MAX_LEN = 200
CHAIN_COUNT = 20


def study_instance(instance_path: Path, routes_per_instance: int, seed: int) -> dict:
    loaded = load_instance(instance_path)
    inst = to_core(loaded)
    rng = random.Random(seed)
    corpus = route_corpus(instance_path, loaded.instance.num_customers, rng, routes_per_instance)

    out: dict = {"instance": str(instance_path), "routes": len(corpus), "modes": {}}
    profiles0 = [_core.fold_profile(inst, r, 0) for r in corpus]
    for mode, name in MODES.items():
        total_bp = 0
        max_prefix = 0
        sum_max_prefix = 0
        feasible = 0
        dstar_bitwise = 0
        dstar_diff_max = 0.0
        dstar_diff_over_1e9 = 0
        feas_flips = 0
        value_diff_max = 0.0
        for r, p0 in zip(corpus, profiles0):
            pm = p0 if mode == 0 else _core.fold_profile(inst, r, mode)
            total_bp += len(pm["xs"])
            if pm["prefix_bp"]:
                mp = max(pm["prefix_bp"])
                max_prefix = max(max_prefix, mp)
                sum_max_prefix += mp
            if pm["feasible"]:
                feasible += 1
            if pm["feasible"] != p0["feasible"]:
                feas_flips += 1
                continue
            if not p0["feasible"]:
                dstar_bitwise += 1
                continue
            if pm["delta_star"] == p0["delta_star"]:
                dstar_bitwise += 1
            else:
                d = abs(pm["delta_star"] - p0["delta_star"])
                dstar_diff_max = max(dstar_diff_max, d)
                if d > 1e-9:
                    dstar_diff_over_1e9 += 1
            if mode != 0:
                seen: set[float] = set()
                for x, y in zip(p0["xs"], p0["ys"]):
                    if x in seen:
                        continue  # steps: only the first value is observable
                    seen.add(x)
                    if pm["xs"][0] <= x <= pm["xs"][-1]:
                        value_diff_max = max(
                            value_diff_max,
                            abs(_core.evaluate_pwlf(pm["xs"], pm["ys"], x) - y),
                        )
        out["modes"][name] = {
            "total_final_bp": total_bp,
            "max_prefix_bp": max_prefix,
            "sum_max_prefix_bp": sum_max_prefix,
            "feasible": feasible,
            "dstar_bitwise": dstar_bitwise,
            "dstar_diff_max": dstar_diff_max,
            "dstar_diff_over_1e9": dstar_diff_over_1e9,
            "feasibility_flips": feas_flips,
            "delta_value_diff_max": value_diff_max,
        }
    return out


def timing_instance(instance_path: Path, num_moves: int, fold_repeats: int, seed: int) -> dict:
    loaded = load_instance(instance_path)
    inst = to_core(loaded)
    bks_path = instance_path.with_name(instance_path.name.replace(".vrp.json", ".bks.Duration.json"))
    routes = [list(map(int, r)) for r in json.loads(bks_path.read_text())["routes"]]
    out: dict = {
        "instance": str(instance_path),
        "num_routes": len(routes),
        "mean_len": sum(len(r) for r in routes) / len(routes),
        "modes": {},
    }
    for mode, name in MODES.items():
        entry: dict = {}
        fb = _core.bench_fold(inst, routes, fold_repeats, mode)
        entry["fold_ns"] = fb["ns_per_fold"]
        entry["fold_checksum"] = fb["checksum"]
        if len(routes) >= 2:
            for method, mname in ((0, "naive"), (1, "bbt"), (2, "lca")):
                r = _core.bench_exchange(inst, routes, num_moves, 8, seed, method, mode)
                entry[f"{mname}_ns"] = r["ns_per_eval"]
                entry[f"{mname}_feasible"] = r["feasible"]
                entry[f"{mname}_checksum"] = r["checksum"]
                if method > 0:
                    entry[f"{mname}_build_ns"] = r["build_ns_total"]
        entry["lca_stored_bp"] = sum(
            _core.LcaTree(inst, r, mode=mode).total_stored_bp for r in routes
        )
        entry["bbt_stored_bp"] = sum(
            _core.RouteTree(inst, r, mode=mode).total_stored_bp for r in routes
        )
        out["modes"][name] = entry
    return out


def chain_instance(instance_path: Path, seed: int) -> dict:
    """Fold long random chains and record the prefix-size growth curve of
    every mode until the chain leaves the horizon (empty function)."""
    loaded = load_instance(instance_path)
    inst = to_core(loaded)
    n = loaded.instance.num_customers
    rng = random.Random(seed)
    chains = []
    for _ in range(CHAIN_COUNT):
        c = [rng.randint(1, n) for _ in range(CHAIN_MAX_LEN)]
        for k in range(1, CHAIN_MAX_LEN):  # no self arcs
            while c[k] == c[k - 1]:
                c[k] = rng.randint(1, n)
        chains.append(c)
    out: dict = {"instance": str(instance_path), "modes": {}}
    for mode, name in MODES.items():
        curves = [_core.fold_profile(inst, c, mode)["prefix_bp"] for c in chains]
        depth = max(len(c) for c in curves)
        mean_curve = []
        max_curve = []
        alive = []
        for k in range(depth):
            vals = [c[k] for c in curves if len(c) > k and c[k] > 0]
            alive.append(len(vals))
            mean_curve.append(sum(vals) / len(vals) if vals else 0.0)
            max_curve.append(max(vals) if vals else 0)
        out["modes"][name] = {
            "max_feasible_prefix": depth,
            "alive_at_prefix": alive,
            "mean_prefix_bp": mean_curve,
            "max_prefix_bp": max_curve,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=int, default=200)
    parser.add_argument("--moves", type=int, default=20000)
    parser.add_argument("--fold-repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="tiny corpus for smoke runs")
    parser.add_argument("--output", type=Path, default=Path("norm_study.json"))
    args = parser.parse_args()

    root = benchmarks_root()
    results: dict = {"accuracy": [], "timing": [], "chains": []}

    picks = DEFAULT_PICKS[:3] if args.quick else DEFAULT_PICKS
    for problem_type, family, size_dir, max_instances in picks:
        directory = root / problem_type / family / size_dir
        if not directory.is_dir():
            continue
        count = 1 if args.quick else max_instances
        for instance_path in sorted(directory.glob("*.vrp.json"))[:count]:
            stats = study_instance(instance_path, args.routes if not args.quick else 40, args.seed)
            stats.update(problem_type=problem_type, family=family, size=size_dir)
            results["accuracy"].append(stats)
            m = stats["modes"]
            print(
                f"[acc] {problem_type}/{family}/{instance_path.stem}: bp "
                + " ".join(
                    f"{name}={m[name]['total_final_bp']}" for name in MODES.values()
                )
                + f" | dstar_eq v2={m['collinear']['dstar_bitwise']}/{stats['routes']}"
                f" v3={m['eps_slope']['dstar_bitwise']}/{stats['routes']}"
                f" | flips v3={m['eps_slope']['feasibility_flips']}"
            )

    if not args.quick:
        for problem_type, family, size_dir, stem in EXTRA_ACCURACY:
            if stem is None:
                found = sorted((root / problem_type / family / size_dir).glob("*.vrp.json"))
                if not found:
                    continue
                path = found[0]
            else:
                path = root / problem_type / family / size_dir / f"{stem}.vrp.json"
                if not path.is_file():
                    continue
            stats = study_instance(path, args.routes, args.seed)
            stats.update(problem_type=problem_type, family=family, size=size_dir)
            results["accuracy"].append(stats)
            m = stats["modes"]
            print(
                f"[acc] {problem_type}/{family}/{path.stem}: bp "
                + " ".join(f"{name}={m[name]['total_final_bp']}" for name in MODES.values())
                + f" | dstar_eq v2={m['collinear']['dstar_bitwise']}/{stats['routes']}"
                f" v3={m['eps_slope']['dstar_bitwise']}/{stats['routes']}"
                f" | flips v3={m['eps_slope']['feasibility_flips']}"
            )

    for problem_type, family, size_dir, stem in BENCH_PICKS:
        path = root / problem_type / family / size_dir / f"{stem}.vrp.json"
        if not path.is_file():
            continue
        stats = timing_instance(path, args.moves if not args.quick else 2000,
                                args.fold_repeats if not args.quick else 5, args.seed)
        stats.update(problem_type=problem_type, family=family, size=size_dir)
        results["timing"].append(stats)
        m = stats["modes"]
        print(
            f"[time] {problem_type}/{family}/{stem}: fold "
            + " ".join(f"{name}={m[name]['fold_ns']/1e3:.1f}us" for name in MODES.values())
        )

    for problem_type, family, size_dir, stem in CHAIN_PICKS:
        directory = root / problem_type / family / size_dir
        if not directory.is_dir():
            continue
        if stem is None:
            candidates = sorted(directory.glob("*.vrp.json"))
            if not candidates:
                continue
            path = candidates[0]
        else:
            path = directory / f"{stem}.vrp.json"
            if not path.is_file():
                continue
        stats = chain_instance(path, args.seed)
        stats.update(problem_type=problem_type, family=family, size=size_dir)
        results["chains"].append(stats)
        m0 = stats["modes"]["none"]
        print(
            f"[chain] {problem_type}/{family}/{path.stem}: max feasible prefix "
            f"{m0['max_feasible_prefix']}, peak bp "
            + " ".join(
                f"{name}={max(stats['modes'][name]['max_prefix_bp'], default=0)}"
                for name in MODES.values()
            )
        )

    args.output.write_text(json.dumps(results, indent=2))
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
