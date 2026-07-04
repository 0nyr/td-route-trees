#include "structures/bbt.h"

#include <algorithm>
#include <stdexcept>

#include "structures/baseline.h"
#include "structures/splice.h"

namespace trt {

using kayros::Instance;
using kayros::Pwlf;
using kayros::PwlfView;
using kayros::RouteEval;
using kayros::compose;
using kayros::identity;
using kayros::make_theta;
using kayros::min_shifted_image;
using kayros::view;

void RouteTree::build(std::vector<Pwlf> leaves) {
    if (leaves.empty()) throw std::invalid_argument("no leaves");
    leaves_ = std::move(leaves);
    nodes_.clear();
    nodes_.reserve(2 * leaves_.size());
    root_ = build_range(0, static_cast<std::int64_t>(leaves_.size()) - 1);
}

std::int32_t RouteTree::build_range(std::int64_t lo, std::int64_t hi) {
    Node node;
    node.lo = lo;
    node.hi = hi;
    if (lo == hi) {
        node.fn = leaves_[static_cast<std::size_t>(lo)];
    } else {
        // Same association as fold_balanced: mid splits [lo, hi+1) at
        // lo + (hi + 1 - lo) / 2, i.e. left = [lo, mid-1], right = [mid, hi].
        const std::int64_t mid = lo + (hi + 1 - lo) / 2;
        node.left = build_range(lo, mid - 1);
        node.right = build_range(mid, hi);
        const Pwlf& lf = nodes_[static_cast<std::size_t>(node.left)].fn;
        const Pwlf& rf = nodes_[static_cast<std::size_t>(node.right)].fn;
        if (lf.xs.empty() || rf.xs.empty()) {
            node.fn = {};
        } else {
            node.fn = compose(view(rf), view(lf));
        }
    }
    nodes_.push_back(std::move(node));
    return static_cast<std::int32_t>(nodes_.size()) - 1;
}

const Pwlf& RouteTree::root() const {
    if (root_ < 0) throw std::logic_error("tree not built");
    return nodes_[static_cast<std::size_t>(root_)].fn;
}

void RouteTree::cover(std::int32_t index, std::int64_t lo, std::int64_t hi,
                      std::vector<std::int32_t>& out) const {
    const Node& node = nodes_[static_cast<std::size_t>(index)];
    if (lo <= node.lo && node.hi <= hi) {
        out.push_back(index);
        return;
    }
    if (node.left < 0) throw std::logic_error("leaf outside cover range");
    const Node& left = nodes_[static_cast<std::size_t>(node.left)];
    if (lo <= left.hi) cover(node.left, lo, hi, out);
    if (hi > left.hi) cover(node.right, lo, hi, out);
}

Pwlf RouteTree::query(std::int64_t lo, std::int64_t hi) const {
    if (root_ < 0) throw std::logic_error("tree not built");
    if (lo < 0 || hi >= num_leaves() || lo > hi) {
        throw std::invalid_argument("invalid query range");
    }
    if (lo == hi) return leaves_[static_cast<std::size_t>(lo)];
    std::vector<std::int32_t> cover_nodes;
    cover(root_, lo, hi, cover_nodes);
    // cover_nodes are in leaf-index order n_0 .. n_{k-1}; the result is
    // n_{k-1} ∘ ... ∘ n_0. Fold each side smallest-first (node sizes increase
    // toward the split point) so total work stays O(m̄ p): left part
    // left-to-right, right part right-to-left, then one final compose.
    std::size_t split = 0;
    std::int64_t best = -1;
    for (std::size_t s = 0; s < cover_nodes.size(); ++s) {
        const Node& node = nodes_[static_cast<std::size_t>(cover_nodes[s])];
        const std::int64_t size = node.hi - node.lo;
        if (size >= best) {
            best = size;
            split = s;
        }
    }
    Pwlf left_agg = nodes_[static_cast<std::size_t>(cover_nodes[0])].fn;
    for (std::size_t s = 1; s <= split; ++s) {
        if (left_agg.xs.empty()) return left_agg;
        left_agg = compose(view(nodes_[static_cast<std::size_t>(cover_nodes[s])].fn),
                           view(left_agg));
    }
    if (split + 1 == cover_nodes.size()) return left_agg;
    Pwlf right_agg = nodes_[static_cast<std::size_t>(cover_nodes.back())].fn;
    for (std::size_t s = cover_nodes.size() - 1; s > split + 1; --s) {
        if (right_agg.xs.empty()) return right_agg;
        right_agg = compose(view(right_agg),
                            view(nodes_[static_cast<std::size_t>(cover_nodes[s - 1])].fn));
    }
    if (left_agg.xs.empty() || right_agg.xs.empty()) return {};
    return compose(view(right_agg), view(left_agg));
}

RouteEval eval_spliced(const Instance& inst, const RouteTree& tree1,
                       const std::vector<std::int32_t>& route1, std::int64_t i1,
                       std::int64_t j1, const RouteTree& tree2,
                       const std::vector<std::int32_t>& route2, std::int64_t i2,
                       std::int64_t j2) {
    return eval_spliced_impl(inst, tree1, route1, i1, j1, tree2, route2, i2, j2);
}

}  // namespace trt
