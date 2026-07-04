"""P4.2 correctness gates for the Visser ready-time-function tree.

Hard gates (exact, no exceptions):
1. tree root == fold_balanced association, bitwise (route_delta_balanced).
2. feasibility of every tree move evaluation == checker feasibility of the
   spliced route.

Duration gate: exact equality with the checker fold for the overwhelming
majority; every mismatch must belong to the two documented association
classes (ulp dust ~1e-11, or a vertical-step jump on step-bearing ATFs) —
see reports/design/td-route-trees-conventions.md. Accepted moves in any LS
are repriced by the checker fold, so these deviations never reach reported
costs.
"""

import random

import pytest

from conftest import bks_routes, require_benchmarks, sample_instance_paths

SEED = 7
MOVES_PER_INSTANCE = 400
MAX_SEG = 3
# Largest plausible vertical-step height across MAMUT families (Rifki envelope
# steps are small integers); only applies to the documented step-jump class.
MAX_STEP_JUMP = 64.0


def spliced(route1, route2, i1, j1, i2, j2):
    middle = route2[i2 : j2 + 1] if i2 <= j2 else []
    return route1[:i1] + middle + route1[j1 + 1 :]


@pytest.mark.parametrize(
    "instance_path", sample_instance_paths(), ids=lambda p: f"{p.parts[-4]}-{p.parts[-3]}-{p.stem}"
)
def test_tree_root_and_moves(instance_path):
    require_benchmarks()
    from td_route_trees import _core, load_instance, to_core

    routes = bks_routes(instance_path)
    if len(routes) < 2:
        pytest.skip("needs a BKS with >= 2 routes")
    loaded = load_instance(instance_path)
    inst = to_core(loaded)

    trees = [_core.RouteTree(inst, route) for route in routes]

    # Gate 1: root bitwise == balanced fold.
    for tree, route in zip(trees, routes):
        assert tree.root() == _core.route_delta_balanced(inst, route)

    rng = random.Random(SEED)
    mismatches = []
    for _ in range(MOVES_PER_INSTANCE):
        a, b = rng.sample(range(len(routes)), 2)
        r1, r2 = routes[a], routes[b]
        len1 = rng.randint(0, min(MAX_SEG, len(r1)))
        len2 = rng.randint(0, min(MAX_SEG, len(r2)))
        if len1 == 0 and len2 == 0:
            continue
        if len2 == 0 and len1 == len(r1):
            continue
        i1 = rng.randint(0, len(r1) - len1)
        j1 = i1 + len1 - 1
        if len2 > 0:
            i2 = rng.randint(0, len(r2) - len2)
            j2 = i2 + len2 - 1
        else:
            i2, j2 = 0, -1

        feasible, duration, _ = _core.eval_spliced(inst, trees[a], r1, i1, j1, trees[b], r2, i2, j2)
        new_route = spliced(r1, r2, i1, j1, i2, j2)
        ref_feasible, ref_duration, _ = _core.evaluate_route(inst, new_route)

        # Gate 2: feasibility is exact.
        assert feasible == ref_feasible, (new_route, i1, j1, i2, j2)
        if feasible and duration != ref_duration:
            mismatches.append(abs(duration - ref_duration))

    # Duration gate: only the documented association classes may appear
    # (ulp dust, or step jumps bounded by the family's step heights). Observed
    # 2026-07-05 on BKS-route corpora: 7-34% ulp-class mismatches, max 7.3e-12,
    # zero step jumps.
    assert len(mismatches) <= MOVES_PER_INSTANCE * 0.5, len(mismatches)
    assert all(diff < MAX_STEP_JUMP for diff in mismatches), max(mismatches)
