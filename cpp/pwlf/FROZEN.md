# Frozen NDCPWLF engine copy

`pwlf.h` / `pwlf.cpp` and the checker-fold reference (`../core/instance.h`, `../core/route_eval.cpp`) are **verbatim frozen copies from kayros**, the canonical home of the engine (bit-identical port of the `mamut_routing_lib.td.pwlf` reference checker).

- Source: https://github.com/0nyr/kayros, commit `67c76d4b242ac554c0b9a11642a695a5f46b45bc` (2026-07-04).
- Do not edit these files here. Engine fixes land in kayros first (with the checker-equivalence suite green), then the copy is refreshed and this commit hash updated.
- `-ffp-contract=off` is mandatory for every target compiling these files (set globally in `CMakeLists.txt`).
