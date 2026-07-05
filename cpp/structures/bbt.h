#pragma once

#include <cstdint>
#include <vector>

#include "core/instance.h"
#include "structures/norm.h"

namespace trt {

// Visser & Spliet 2020 ready-time-function tree: a balanced binary tree over
// the m+1 route leaves (P4.0 memo convention). Internal node = composition of
// its children; the root is delta_r with the same association as
// fold_balanced (mid = lo + (hi - lo) / 2), so root() must agree bitwise with
// fold_balanced(route_leaves(...)).
//
// query(lo, hi) returns the composition of leaves [lo, hi] via the classic
// range decomposition (O(log m) cover nodes, <= 2 per level), folded
// smallest-first from both ends: O(m̄ p) total work (Visser Theorem 8).
//
// Updates are full rebuilds (Visser Section 5.6): call build() again.
class RouteTree {
  public:
    // Norm study: mode != none prunes every internal composition (callers
    // should pass leaves from route_leaves_norm with the same mode).
    void build(std::vector<kayros::Pwlf> leaves,
               NormMode mode = NormMode::none);

    std::int64_t num_leaves() const { return static_cast<std::int64_t>(leaves_.size()); }
    NormMode norm_mode() const { return mode_; }
    // Breakpoints stored across all tree nodes (structure memory metric).
    std::int64_t total_stored_bp() const;
    const kayros::Pwlf& root() const;
    const kayros::Pwlf& leaf(std::int64_t i) const { return leaves_[static_cast<std::size_t>(i)]; }

    // Composition of leaves lo..hi inclusive; empty Pwlf propagates. lo > hi
    // is invalid (callers skip empty ranges).
    kayros::Pwlf query(std::int64_t lo, std::int64_t hi) const;

  private:
    struct Node {
        std::int64_t lo, hi;  // leaf range [lo, hi] inclusive
        std::int32_t left = -1, right = -1;
        kayros::Pwlf fn;
    };
    std::int32_t build_range(std::int64_t lo, std::int64_t hi);
    void cover(std::int32_t node, std::int64_t lo, std::int64_t hi,
               std::vector<std::int32_t>& out) const;

    std::vector<kayros::Pwlf> leaves_;
    std::vector<Node> nodes_;
    std::int32_t root_ = -1;
    NormMode mode_ = NormMode::none;
};

// Exchange-move evaluation on trees (Visser Eq. 7 generalized): the new first
// route r1' = r1[0..i1-1] ++ r2[i2..j2] ++ r1[j1+1..], with empty segments
// encoded as j < i (pure insertion/deletion points). Relocate, swap, 2-opt*
// and insertion are all special cases. Returns the checker-convention
// evaluation of r1' built from tree partials + fresh bridge leaves.
kayros::RouteEval eval_spliced(const kayros::Instance& inst,
                               const RouteTree& tree1,
                               const std::vector<std::int32_t>& route1,
                               std::int64_t i1, std::int64_t j1,
                               const RouteTree& tree2,
                               const std::vector<std::int32_t>& route2,
                               std::int64_t i2, std::int64_t j2);

}  // namespace trt
