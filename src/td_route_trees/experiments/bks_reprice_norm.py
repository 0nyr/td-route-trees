"""Reprice every stored BKS under the four norm engine variants.

Quantifies the "re-evaluate all BKS" cost of evolving the checker: for each
``*.bks.Duration.json`` in the benchmarks tree, each route is folded under
modes none/safe/collinear/eps_slope and the solution total is compared to
the stored duration (bitwise). V0 must reproduce the store exactly (sanity);
any V1 deviation falsifies the safe-neutrality claim; V2/V3 deviations price
the semantic switch.

Usage:
    python -m td_route_trees.experiments.bks_reprice_norm --output bks_reprice_norm.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from td_route_trees import _core, load_instance, to_core
from td_route_trees.experiments.fold_order import benchmarks_root

MODES = {0: "none", 1: "safe", 2: "collinear", 3: "eps_slope"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("bks_reprice_norm.json"))
    args = parser.parse_args()

    root = benchmarks_root()
    totals = {
        name: {"solutions": 0, "exact": 0, "changed": 0, "max_abs_diff": 0.0,
               "max_rel_diff": 0.0, "feasibility_flips": 0, "worst": None}
        for name in MODES.values()
    }
    families: dict[str, dict] = {}

    bks_paths = sorted(root.rglob("*.bks.Duration.json"))
    print(f"{len(bks_paths)} BKS files under {root}")
    for k, bks_path in enumerate(bks_paths):
        instance_path = bks_path.with_name(bks_path.name.replace(".bks.Duration.json", ".vrp.json"))
        if not instance_path.is_file():
            continue
        data = json.loads(bks_path.read_text())
        stored = float(data["duration"])
        routes = [list(map(int, r)) for r in data["routes"]]
        loaded = load_instance(instance_path)
        inst = to_core(loaded)
        fam_key = "/".join(bks_path.relative_to(root).parts[:2])
        fam = families.setdefault(
            fam_key,
            {name: {"solutions": 0, "exact": 0, "changed": 0, "max_abs_diff": 0.0,
                    "feasibility_flips": 0} for name in MODES.values()},
        )
        for mode, name in MODES.items():
            total = 0.0
            flip = False
            for route in routes:
                p = _core.fold_profile(inst, route, mode)
                if not p["feasible"]:
                    flip = True
                    break
                total += p["delta_star"]
            t = totals[name]
            f = fam[name]
            t["solutions"] += 1
            f["solutions"] += 1
            if flip:
                t["feasibility_flips"] += 1
                f["feasibility_flips"] += 1
                continue
            if total == stored:
                t["exact"] += 1
                f["exact"] += 1
            else:
                t["changed"] += 1
                f["changed"] += 1
                diff = abs(total - stored)
                f["max_abs_diff"] = max(f["max_abs_diff"], diff)
                if diff > t["max_abs_diff"]:
                    t["max_abs_diff"] = diff
                    t["worst"] = str(bks_path.relative_to(root))
                if stored:
                    t["max_rel_diff"] = max(t["max_rel_diff"], diff / abs(stored))
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(bks_paths)} done")

    print(json.dumps(totals, indent=2))
    args.output.write_text(json.dumps({"totals": totals, "families": families}, indent=2))
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
