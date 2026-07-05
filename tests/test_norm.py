"""Gates for the normalization study (norm-vs-no-norm checker question).

V1 ``safe`` (interior points of exactly-flat / exactly-vertical runs) claims
full evaluation neutrality: same domain, same values at every retained and
pruned abscissa, same Delta* and argmin, same feasibility — bitwise. That
claim is gated here, on real corpora, at every level (fold, BBT, LCA).

V2 ``collinear`` and V3 ``eps_slope`` make no neutrality claim (V2 drifts by
ulps downstream, V3 is lossy by construction); they are measured by the
experiment driver, not gated — only structural invariants are checked.
"""

from __future__ import annotations

import random

import pytest

from conftest import bks_routes, random_routes, require_benchmarks, sample_instance_paths
from td_route_trees import _core, load_instance, to_core

MODES = {"none": 0, "safe": 1, "collinear": 2, "eps_slope": 3}
SEED = 11


def test_prune_toy_cases() -> None:
    # Flat run: interior points dropped, endpoints kept.
    xs, ys = _core.prune([0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0, 6.0], MODES["safe"])
    assert xs == [0.0, 3.0, 4.0] and ys == [5.0, 5.0, 6.0]
    # Vertical run: interior values dropped, first and last kept.
    xs, ys = _core.prune([0.0, 1.0, 1.0, 1.0, 2.0], [0.0, 1.0, 2.0, 3.0, 4.0], MODES["safe"])
    assert xs == [0.0, 1.0, 1.0, 2.0] and ys == [0.0, 1.0, 3.0, 4.0]
    # Exactly-collinear interior point: kept by safe, dropped by collinear.
    xs, ys = _core.prune([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], MODES["safe"])
    assert len(xs) == 3
    xs, ys = _core.prune([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], MODES["collinear"])
    assert xs == [0.0, 2.0] and ys == [0.0, 2.0]
    # Non-collinear kink survives collinear pruning.
    xs, ys = _core.prune([0.0, 1.0, 2.0], [0.0, 1.0, 4.0], MODES["collinear"])
    assert len(xs) == 3
    # eps_slope merges slopes closer than 1e-6 (lossy: the kink vanishes).
    xs, ys = _core.prune([0.0, 1.0, 2.0], [0.0, 1.0, 2.0 + 1e-7], MODES["eps_slope"])
    assert len(xs) == 2


@pytest.mark.parametrize("mode_name", ["safe", "collinear", "eps_slope"])
def test_prune_keeps_domain_and_monotonicity(mode_name: str) -> None:
    require_benchmarks()
    mode = MODES[mode_name]
    rng = random.Random(SEED)
    for path in sample_instance_paths():
        loaded = load_instance(path)
        inst = to_core(loaded)
        for route in random_routes(loaded.instance.num_customers, rng, count=40):
            p0 = _core.fold_profile(inst, route, 0)
            pm = _core.fold_profile(inst, route, mode)
            if not p0["xs"]:
                assert not pm["xs"]
                continue
            assert pm["xs"], f"{mode_name} lost feasibility on {path.name}"
            if mode_name == "safe":
                assert pm["xs"][0] == p0["xs"][0]
                assert pm["xs"][-1] == p0["xs"][-1]
            else:
                # V2/V3 domain endpoints may drift by ulps through recursive
                # composition (interpolated preimages over pruned operands).
                assert pm["xs"][0] == pytest.approx(p0["xs"][0], rel=1e-9, abs=1e-6)
                assert pm["xs"][-1] == pytest.approx(p0["xs"][-1], rel=1e-9, abs=1e-6)
            xs, ys = pm["xs"], pm["ys"]
            assert all(a <= b for a, b in zip(xs, xs[1:]))
            assert all(a <= b for a, b in zip(ys, ys[1:]))


def test_safe_mode_is_evaluation_neutral() -> None:
    """V1 gate: pruned delta evaluates bitwise-identically at every V0
    breakpoint, and Delta* / argmin / feasibility match bitwise."""
    require_benchmarks()
    rng = random.Random(SEED)
    for path in sample_instance_paths():
        loaded = load_instance(path)
        inst = to_core(loaded)
        corpus = bks_routes(path) + random_routes(loaded.instance.num_customers, rng, count=60)
        for route in corpus:
            p0 = _core.fold_profile(inst, route, 0)
            p1 = _core.fold_profile(inst, route, MODES["safe"])
            assert p1["feasible"] == p0["feasible"]
            if not p0["feasible"]:
                continue
            assert p1["delta_star"] == p0["delta_star"]
            assert p1["argmin_x"] == p0["argmin_x"]
            first_y_at = {}
            for x, y in zip(p0["xs"], p0["ys"]):
                first_y_at.setdefault(x, y)
            for x, y in first_y_at.items():
                # The checker evaluate returns the first (smallest) value at a
                # duplicated abscissa, so only that value is observable.
                assert _core.evaluate_pwlf(p1["xs"], p1["ys"], x) == y


def test_safe_mode_neutral_in_structures() -> None:
    """V1 inside BBT and LCA: Delta* of every (lo, hi) query matches the
    mode-none tree bitwise, and stored size never grows."""
    require_benchmarks()
    for path in sample_instance_paths():
        loaded = load_instance(path)
        inst = to_core(loaded)
        for route in bks_routes(path):
            t0 = _core.LcaTree(inst, route, mode=0)
            t1 = _core.LcaTree(inst, route, mode=MODES["safe"])
            b0 = _core.RouteTree(inst, route, mode=0)
            b1 = _core.RouteTree(inst, route, mode=MODES["safe"])
            m = t0.num_leaves
            for lo in range(m):
                for hi in range(lo, m):
                    xs0, ys0 = t0.query(lo, hi)
                    xs1, ys1 = t1.query(lo, hi)
                    assert bool(xs0) == bool(xs1)
                    if xs0:
                        assert _core.min_shifted_image(xs0, ys0) == _core.min_shifted_image(xs1, ys1)
                    xs0, ys0 = b0.query(lo, hi)
                    xs1, ys1 = b1.query(lo, hi)
                    assert bool(xs0) == bool(xs1)
                    if xs0:
                        assert _core.min_shifted_image(xs0, ys0) == _core.min_shifted_image(xs1, ys1)
            assert t1.total_stored_bp <= t0.total_stored_bp
            assert b1.total_stored_bp <= b0.total_stored_bp
