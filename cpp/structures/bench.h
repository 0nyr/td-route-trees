#pragma once

#include <cstdint>
#include <vector>

#include "core/instance.h"

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
BenchResult bench_exchange(const kayros::Instance& inst,
                           const std::vector<std::vector<std::int32_t>>& routes,
                           std::int64_t num_moves, std::int64_t max_seg,
                           std::uint64_t seed, int method);

}  // namespace trt
