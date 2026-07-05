"""P4.4/P4.5 axes: update cost (BBT rebuild vs LCA localized) and prefix reuse.

Update axis (P4.5): after an accepted move, the Visser BBT rebuilds the whole
tree while the Blauth LCA-BST recomputes only the stored compositions
straddling the changed leaf. Both methods apply the same seeded sequence of
leaf replacements; feasible counts must match.

Scan axis (P4.4, time-boxed): scoring the insertion of many candidates at
every position of a route either queries the tree prefix per evaluation or
advances a shared left-fold prefix once per position (associations differ,
so checksums agree only to ulp class; feasible counts must match).

Usage:
    python -m td_route_trees.experiments.update_scan --output update_scan.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from td_route_trees import _core, load_instance, to_core
from td_route_trees.experiments.fold_order import benchmarks_root

PICKS: list[tuple[str, str, str, str]] = [
    ("TDVRPTW", "Dabia2013", "n=100", "R211"),
    ("TDVRPTW", "Dabia2013", "n=100", "RC203"),
    ("TDVRPTW", "Dabia2013", "n=100", "R112"),
    ("TDVRP", "Dabia2013", "n=100", "R211"),
    ("TDVRPTW", "Rifki2020", "n=30", "Rifki-10"),
    ("TDVRPTW", "Ari2018", "n=30", "Ari-A10-pB-d70-w50"),
    ("TDVRPTW", "Vu2020", "n=99", "Vu-A1-pA-d70-w150"),
    ("TDVRP", "Vu2020", "n=99", "Vu-A1-pA-d70-w150"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=20000)
    parser.add_argument("--candidates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("update_scan.json"))
    args = parser.parse_args()

    root = benchmarks_root()
    results: list[dict] = []
    for problem_type, family, size_dir, stem in PICKS:
        path = root / problem_type / family / size_dir / f"{stem}.vrp.json"
        if not path.is_file():
            candidates = sorted((root / problem_type / family / size_dir).glob("*.vrp.json"))
            if not candidates:
                continue
            path = candidates[0]
        loaded = load_instance(path)
        inst = to_core(loaded)
        bks = json.loads(
            path.with_name(path.name.replace(".vrp.json", ".bks.Duration.json")).read_text()
        )
        route = max((list(map(int, r)) for r in bks["routes"]), key=len)

        entry: dict = {
            "instance": str(path),
            "problem_type": problem_type,
            "family": family,
            "route_len": len(route),
        }
        up_bbt = _core.bench_update(inst, route, args.updates, args.seed, 1)
        up_lca = _core.bench_update(inst, route, args.updates, args.seed, 2)
        entry["update_bbt"] = up_bbt
        entry["update_lca"] = up_lca
        assert up_bbt["feasible"] == up_lca["feasible"], entry["instance"]

        sc_tree = _core.bench_insertion_scan(inst, route, args.candidates, args.seed, 0)
        sc_incr = _core.bench_insertion_scan(inst, route, args.candidates, args.seed, 1)
        entry["scan_tree"] = sc_tree
        entry["scan_incremental"] = sc_incr
        assert sc_tree["feasible"] == sc_incr["feasible"], entry["instance"]

        results.append(entry)
        print(
            f"{problem_type}/{family}/{path.stem} (m={len(route)}): "
            f"update bbt={up_bbt['ns_per_eval']/1e3:.2f}us lca={up_lca['ns_per_eval']/1e3:.2f}us "
            f"(x{up_bbt['ns_per_eval']/up_lca['ns_per_eval']:.1f}) | "
            f"scan tree={sc_tree['ns_per_eval']/1e3:.2f}us incr={sc_incr['ns_per_eval']/1e3:.2f}us "
            f"(x{sc_tree['ns_per_eval']/sc_incr['ns_per_eval']:.2f}) | "
            f"feas {up_lca['feasible']}/{sc_incr['feasible']}"
        )

    args.output.write_text(json.dumps(results, indent=2))
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
