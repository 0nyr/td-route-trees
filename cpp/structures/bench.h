#pragma once

#include <cstdint>
#include <vector>

#include "core/instance.h"
#include "structures/norm.h"

namespace trt {

struct BenchResult {
    double ns_per_eval = 0.0;
    std::int64_t evals = 0;
    std::int64_t feasible = 0;
    double checksum = 0.0;      // sum of feasible durations (guards against DCE / divergence)
    double build_ns_total = 0.0;  // structure construction time (0 for naive)
};

// Move-evaluation throughput on random exchange moves (segment lengths
// 0..max_seg on each side, not both empty) between random route pairs.
// method: 0 = naive full recomposition (checker fold on the spliced route),
//         1 = Visser tree (prebuilt per route; build time reported separately),
//         2 = Blauth LCA-BST (prebuilt per route; build time reported separately).
// The same seed generates the same move sequence for every method, so
// checksums are comparable across methods (up to the documented ulp/step
// association classes).
//
// mode != none runs the whole evaluation under the normalizing engine:
// naive switches from evaluate_route to the pruned leaf fold, trees are
// built with pruned leaves and prune every stored/queried composition.
BenchResult bench_exchange(const kayros::Instance& inst,
                           const std::vector<std::vector<std::int32_t>>& routes,
                           std::int64_t num_moves, std::int64_t max_seg,
                           std::uint64_t seed, int method,
                           NormMode mode = NormMode::none);

// P4.5 update-cost axis: apply `num_updates` random single-leaf replacements
// (a fresh bridge leaf for a random customer at that position).
// method: 1 = Visser BBT full rebuild, 2 = LCA localized update_leaf.
// checksum = sum of root/full-query Delta* after each update (forces work).
BenchResult bench_update(const kayros::Instance& inst,
                         const std::vector<std::int32_t>& route,
                         std::int64_t num_updates, std::uint64_t seed,
                         int method);

// P4.4 prefix-reuse axis: score inserting each of `num_candidates` random
// customers at every position of the route.
// method: 0 = per-position tree evaluation (eval_spliced, 3 composes each),
//         1 = incremental left-fold prefix shared across candidates
//             (1 compose to advance the prefix + 2 per candidate).
// Association differs between methods (balanced vs left), so checksums agree
// only up to the documented ulp classes; feasible counts must match.
BenchResult bench_insertion_scan(const kayros::Instance& inst,
                                 const std::vector<std::int32_t>& route,
                                 std::int64_t num_candidates,
                                 std::uint64_t seed, int method);

}  // namespace trt
