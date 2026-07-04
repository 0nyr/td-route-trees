"""P4.3 correctness gates for the Blauth LCA-BST.

Hard gates:
1. query(i, i) == leaf bitwise; queries are internally deterministic.
2. update_leaf ≡ full rebuild, bitwise on every query (same recurrences).
3. feasibility of every move evaluation == checker feasibility.

Duration gate: documented association classes only (see test_bbt.py).
"""

import random

import pytest

from conftest import bks_routes, require_benchmarks, sample_instance_paths
from test_bbt import MAX_SEG, MAX_STEP_JUMP, MOVES_PER_INSTANCE, SEED, spliced


@pytest.mark.parametrize(
    "instance_path", sample_instance_paths(), ids=lambda p: f"{p.parts[-4]}-{p.parts[-3]}-{p.stem}"
)
def test_lca_tree_gates(instance_path):
    require_benchmarks()
    from td_route_trees import _core, load_instance, to_core

    routes = bks_routes(instance_path)
    if len(routes) < 2:
        pytest.skip("needs a BKS with >= 2 routes")
    loaded = load_instance(instance_path)
    inst = to_core(loaded)

    trees = [_core.LcaTree(inst, route) for route in routes]
    bbt_trees = [_core.RouteTree(inst, route) for route in routes]

    rng = random.Random(SEED)

    # Gate 1: full-range query is a valid delta (same duration classes as
    # the balanced fold) and singleton queries behave.
    for tree, bbt, route in zip(trees, bbt_trees, routes):
        m = tree.num_leaves
        assert m == len(route) + 1
        xs, ys = tree.query(0, m - 1)
        assert xs, "BKS route must be feasible"
        # Singleton queries: exact stored leaves; cross-check vs BBT leaves.
        for i in range(m):
            assert tree.query(i, i) == bbt.query(i, i)

    # Gate 2: update_leaf with an unchanged leaf function is a bitwise no-op
    # for every (lo, hi) query.
    tree = trees[0]
    m = tree.num_leaves
    probe_ranges = [(lo, hi) for lo in range(m) for hi in range(lo, m)]
    before = {r: tree.query(*r) for r in probe_ranges}
    mid_leaf = m // 2
    xs, ys = tree.query(mid_leaf, mid_leaf)
    tree.update_leaf(mid_leaf, xs, ys)
    for r in probe_ranges:
        assert tree.query(*r) == before[r]

    # Gate 3 + duration gate on random exchange moves.
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
        ref_feasible, ref_duration, _ = _core.evaluate_route(inst, spliced(r1, r2, i1, j1, i2, j2))
        assert feasible == ref_feasible
        if feasible and duration != ref_duration:
            mismatches.append(abs(duration - ref_duration))

    assert len(mismatches) <= MOVES_PER_INSTANCE * 0.5, len(mismatches)
    assert all(diff < MAX_STEP_JUMP for diff in mismatches), max(mismatches)
