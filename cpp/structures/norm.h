#pragma once

#include <cstdint>
#include <vector>

#include "core/instance.h"

namespace trt {

// Normalization study (norm-vs-no-norm checker question, 2026-07-05).
//
// The checker's compose only drops exact duplicate points; composed route
// functions therefore carry redundant breakpoints (theta plateaus are the
// main source). These modes prune them as a post-pass after every compose:
//
//   none      — V0, the checker status quo (prune is the identity).
//   safe      — V1, drop interior points of exactly-flat runs (equal
//               consecutive ys) and exactly-vertical runs (equal consecutive
//               xs). Claim (gate-tested): evaluation, feasibility, Delta*
//               and its earliest argmin are all bitwise unchanged.
//   collinear — V2, additionally drop a point whose stored y bitwise-equals
//               the chord interpolation of its kept neighbours (the exact
//               evaluate() formula). Neutral at the pruned point itself, but
//               downstream compose interpolation can drift by ulps — same
//               class as the fold-order association differences.
//   eps_slope — V3, faithful port of pwlf_compare's normalized_add predicate
//               (paper0 visser+nor): drop when adjacent slopes differ by
//               less than goc EPS = 1e-6; on ~flat runs the *older* y wins.
//               Lossy by construction; measured here to quantify the drift,
//               not proposed as a checker candidate.
//   safe_int  — V4, same *output* as safe but with the dedup folded into the
//               compose emit loop instead of a post-pass: the actual
//               checker-evolution candidate, timed without copy overhead.
enum class NormMode : int {
    none = 0,
    safe = 1,
    collinear = 2,
    eps_slope = 3,
    safe_int = 4,
};

// In-place pruning pass; first breakpoint always kept, last always kept
// (eps_slope may rewrite its y, faithful to pwlf_compare). safe_int prunes
// exactly like safe.
void prune_inplace(kayros::Pwlf& f, NormMode mode);

// The frozen checker compose with the safe dedup integrated into emit:
// bitwise-identical output to compose + prune(safe).
kayros::Pwlf compose_safe_integrated(kayros::PwlfView f, kayros::PwlfView g);

inline kayros::Pwlf compose_norm(kayros::PwlfView f, kayros::PwlfView g,
                                 NormMode mode) {
    if (mode == NormMode::safe_int) return compose_safe_integrated(f, g);
    kayros::Pwlf h = kayros::compose(f, g);
    prune_inplace(h, mode);
    return h;
}

// route_leaves with every leaf pruned — what a "normalizing engine" would
// hand to any evaluator.
std::vector<kayros::Pwlf> route_leaves_norm(const kayros::Instance& inst,
                                            const std::int32_t* route,
                                            std::int64_t len, NormMode mode);

// Left fold pruning the accumulator after every compose (the normalizing
// engine's sequential route evaluation).
kayros::Pwlf fold_left_norm(const std::vector<kayros::Pwlf>& leaves,
                            NormMode mode);

struct FoldProfile {
    bool feasible = false;
    double delta_star = 0.0;
    double argmin_x = 0.0;
    std::vector<std::int64_t> prefix_bp;  // |acc| after each fold step
    kayros::Pwlf delta;                   // final route function
};

FoldProfile fold_profile(const kayros::Instance& inst,
                         const std::int32_t* route, std::int64_t len,
                         NormMode mode);

struct FoldBench {
    double ns_per_fold = 0.0;
    std::int64_t folds = 0;
    std::int64_t feasible = 0;
    double checksum = 0.0;  // sum of feasible Delta* (guards against DCE)
};

// Repeated sequential evaluation (leaves + fold, both under `mode`) over the
// route set: the naive/checker-engine timing axis of the norm study.
FoldBench bench_fold(const kayros::Instance& inst,
                     const std::vector<std::vector<std::int32_t>>& routes,
                     std::int64_t repeats, NormMode mode);

}  // namespace trt
