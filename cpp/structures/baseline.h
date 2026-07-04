#pragma once

#include <cstdint>
#include <vector>

#include "core/instance.h"

namespace trt {

// Leaf segment functions per the P4.0 conventions memo
// (tdvrptw-workspace/reports/design/td-route-trees-conventions.md), for a
// route r = (o, r0, ..., r_{m-1}, o):
//   L_0     = theta_{r0} ∘ alpha_{o,r0} ∘ identity(depot departure window)
//   L_i     = theta_{r[i]} ∘ alpha_{r[i-1],r[i]}            (1 <= i <= m-1)
//   L_m     = [identity(0, L_depot) ∘] alpha_{r[m-1],o}
// so that delta_r = L_m ∘ ... ∘ L_0, with m+1 leaves in route order.
//
// TDVRP thetas are service shifts over the incoming arc's full image; the
// checker instead shifts over the accumulated image at that leg. Value-equal
// in exact arithmetic, but the interpolation endpoints differ — this is a
// deliberate, experiment-covered difference (see fold_order experiment).
std::vector<kayros::Pwlf> route_leaves(const kayros::Instance& inst,
                                       const std::int32_t* route,
                                       std::int64_t len);

// acc <- L_i ∘ acc for i = 0..m: the checker's association applied to
// precomposed leaves.
kayros::Pwlf fold_left(const std::vector<kayros::Pwlf>& leaves);

// Balanced association: fold(lo, hi) = fold(mid, hi) ∘ fold(lo, mid) — the
// composition order a balanced binary tree of the leaves produces.
kayros::Pwlf fold_balanced(const std::vector<kayros::Pwlf>& leaves);

}  // namespace trt
