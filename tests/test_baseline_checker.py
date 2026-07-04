"""P4.1 correctness anchor: the frozen fold in this repo == the lib checker.

For every sampled instance and a corpus of random + BKS routes, the compiled
`route_delta_checker` / `evaluate_route` must agree exactly (breakpoint lists
and doubles, no epsilons) with `mamut_routing_lib.td.checker`.
"""

import random

import pytest

from conftest import bks_routes, random_routes, require_benchmarks, sample_instance_paths

SEED = 42
RANDOM_ROUTES_PER_INSTANCE = 30


@pytest.mark.parametrize(
    "instance_path", sample_instance_paths(), ids=lambda p: f"{p.parts[-4]}-{p.parts[-3]}-{p.stem}"
)
def test_frozen_fold_matches_checker(instance_path):
    require_benchmarks()
    from mamut_routing_lib.td.checker import (
        compute_route_duration,
        compute_route_ready_time_function,
    )

    from td_route_trees import _core, load_instance, to_core

    loaded = load_instance(instance_path)
    inst = to_core(loaded)
    rng = random.Random(SEED)
    corpus = bks_routes(instance_path) + random_routes(
        loaded.instance.num_customers, rng, RANDOM_ROUTES_PER_INSTANCE
    )
    assert corpus

    checked_feasible = 0
    for route in corpus:
        reference = compute_route_ready_time_function(loaded.instance, loaded.atfs, route)
        xs, ys = _core.route_delta_checker(inst, route)
        assert xs == list(reference.xs)
        assert ys == list(reference.ys)

        evaluation = compute_route_duration(loaded.instance, loaded.atfs, route)
        feasible, duration, departure = _core.evaluate_route(inst, route)
        assert feasible == evaluation.feasible
        if feasible:
            checked_feasible += 1
            assert duration == evaluation.duration
            assert departure == evaluation.departure_time
    # The corpus must exercise the feasible path (BKS routes guarantee it
    # when present; Dabia n=25 randoms also hit feasible short routes).
    if bks_routes(instance_path):
        assert checked_feasible > 0
