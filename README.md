# td-route-trees

Study repository for **fast move evaluation in duration-minimization time-dependent vehicle routing** (TDVRPTW / TDVRP): which data structure should power the KAYROS TD local-search layer?

Candidates, implemented on the same exact NDCPWLF engine and raced on the canonical [MAMUT-routing](https://github.com/ANR-MAMUT/MAMUT-routing) TD benchmark families:

- **P4.1 baseline** — naive full-route recomposition (checker-identical left fold).
- **P4.2** — Visser & Spliet 2020 ready-time-function tree (balanced range decomposition, `O(m̄p)` partial-function queries, full rebuild on change).
- **P4.3** — Blauth et al. 2024 LCA-BST (ancestor–descendant compositions, one-compose queries, `O(n)`-compose localized updates).
- **P4.4** — novel prefix/suffix composition-reuse techniques (time-boxed).

Design conventions (function orientation, TW clamping, FIFO, association-order protocol) are fixed in the workspace memo `reports/design/td-route-trees-conventions.md`; the decision memo for KAYROS is the P4.5 deliverable.

## Engine

`cpp/pwlf/` is a **frozen copy** of the kayros NDCPWLF engine (bit-identical port of the `mamut_routing_lib.td` Duration checker) — see `cpp/pwlf/FROZEN.md` for the recorded commit. The checker remains the referee: every duration reported by any structure is validated against `check_td_solution` pricing in the test suite. Exact IEEE-754 doubles, no epsilons, `-ffp-contract=off`.

## Install & test

```sh
nix develop path:./nix-dev            # or any shell with cmake/ninja/gcc13+/uv
uv venv && source .venv/bin/activate
uv pip install -e . --group dev       # pulls mamut-routing-lib @ git+td
export MAMUT_ROUTING_BENCHMARKS_ROOT=/path/to/MAMUT-routing/benchmarks
pytest
python -m td_route_trees.experiments.fold_order --routes 200
```

## License

MIT.
