#include "structures/bench.h"

#include <chrono>
#include <random>
#include <stdexcept>

#include "structures/baseline.h"
#include "structures/bbt.h"
#include "structures/lca_bst.h"
#include "structures/splice.h"

namespace trt {

using Clock = std::chrono::steady_clock;
using kayros::Instance;
using kayros::RouteEval;

namespace {

struct Move {
    std::size_t r1, r2;
    std::int64_t i1, j1, i2, j2;
};

std::vector<Move> sample_moves(const std::vector<std::vector<std::int32_t>>& routes,
                               std::int64_t num_moves, std::int64_t max_seg,
                               std::uint64_t seed) {
    if (routes.size() < 2) throw std::invalid_argument("need >= 2 routes");
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> pick_route(0, routes.size() - 1);
    std::vector<Move> moves;
    moves.reserve(static_cast<std::size_t>(num_moves));
    while (static_cast<std::int64_t>(moves.size()) < num_moves) {
        Move mv;
        mv.r1 = pick_route(rng);
        do {
            mv.r2 = pick_route(rng);
        } while (mv.r2 == mv.r1);
        const std::int64_t m1 = static_cast<std::int64_t>(routes[mv.r1].size());
        const std::int64_t m2 = static_cast<std::int64_t>(routes[mv.r2].size());
        std::uniform_int_distribution<std::int64_t> len_dist(0, max_seg);
        const std::int64_t len1 = std::min(len_dist(rng), m1);
        const std::int64_t len2 = std::min(len_dist(rng), m2);
        if (len2 == 0 && len1 == m1) continue;  // would empty route 1
        std::uniform_int_distribution<std::int64_t> pos1(0, m1 - len1);
        std::uniform_int_distribution<std::int64_t> pos2(0, m2 - (len2 > 0 ? len2 : 0));
        mv.i1 = pos1(rng);
        mv.j1 = mv.i1 + len1 - 1;  // len1 == 0 -> j1 = i1 - 1 (insertion point)
        if (len2 > 0) {
            mv.i2 = pos2(rng);
            mv.j2 = mv.i2 + len2 - 1;
        } else {
            if (len1 == 0) continue;  // no-op move
            mv.i2 = 0;
            mv.j2 = -1;
        }
        moves.push_back(mv);
    }
    return moves;
}

}  // namespace

BenchResult bench_exchange(const Instance& inst,
                           const std::vector<std::vector<std::int32_t>>& routes,
                           std::int64_t num_moves, std::int64_t max_seg,
                           std::uint64_t seed, int method, NormMode mode) {
    const std::vector<Move> moves = sample_moves(routes, num_moves, max_seg, seed);
    BenchResult result;
    result.evals = static_cast<std::int64_t>(moves.size());

    std::vector<RouteTree> trees;
    std::vector<LcaTree> lca_trees;
    if (method == 1 || method == 2) {
        const auto t0 = Clock::now();
        if (method == 1) trees.resize(routes.size());
        if (method == 2) lca_trees.resize(routes.size());
        for (std::size_t r = 0; r < routes.size(); ++r) {
            auto leaves = route_leaves_norm(
                inst, routes[r].data(),
                static_cast<std::int64_t>(routes[r].size()), mode);
            if (method == 1) {
                trees[r].build(std::move(leaves), mode);
            } else {
                lca_trees[r].build(std::move(leaves), mode);
            }
        }
        result.build_ns_total =
            std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
    }

    const auto t0 = Clock::now();
    for (const Move& mv : moves) {
        RouteEval eval;
        if (method == 0) {
            std::vector<std::int32_t> spliced;
            const auto& r1 = routes[mv.r1];
            const auto& r2 = routes[mv.r2];
            spliced.reserve(r1.size() + r2.size());
            spliced.insert(spliced.end(), r1.begin(), r1.begin() + mv.i1);
            if (mv.i2 <= mv.j2) {
                spliced.insert(spliced.end(), r2.begin() + mv.i2, r2.begin() + mv.j2 + 1);
            }
            spliced.insert(spliced.end(), r1.begin() + mv.j1 + 1, r1.end());
            if (spliced.empty()) continue;
            if (mode == NormMode::none) {
                eval = kayros::evaluate_route(
                    inst, spliced.data(),
                    static_cast<std::int64_t>(spliced.size()));
            } else {
                const auto leaves = route_leaves_norm(
                    inst, spliced.data(),
                    static_cast<std::int64_t>(spliced.size()), mode);
                const kayros::Pwlf delta = fold_left_norm(leaves, mode);
                if (delta.xs.empty()) {
                    eval = {false, 0.0, 0.0};
                } else {
                    const kayros::MinShift s =
                        kayros::min_shifted_image(kayros::view(delta));
                    eval = {true, s.value, s.argmin_x};
                }
            }
        } else if (method == 1) {
            eval = eval_spliced(inst, trees[mv.r1], routes[mv.r1], mv.i1, mv.j1,
                                trees[mv.r2], routes[mv.r2], mv.i2, mv.j2);
        } else {
            eval = eval_spliced_impl(inst, lca_trees[mv.r1], routes[mv.r1], mv.i1,
                                     mv.j1, lca_trees[mv.r2], routes[mv.r2], mv.i2,
                                     mv.j2);
        }
        if (eval.feasible) {
            result.feasible += 1;
            result.checksum += eval.duration;
        }
    }
    const double total_ns =
        std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
    result.ns_per_eval = total_ns / static_cast<double>(result.evals);
    return result;
}

BenchResult bench_update(const Instance& inst,
                         const std::vector<std::int32_t>& route,
                         std::int64_t num_updates, std::uint64_t seed,
                         int method) {
    using kayros::Pwlf;
    const std::int64_t m = static_cast<std::int64_t>(route.size());
    if (m < 2) throw std::invalid_argument("route too short for updates");
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::int64_t> pick_pos(1, m - 1);
    std::uniform_int_distribution<std::int32_t> pick_cust(1, inst.num_customers);

    std::vector<Pwlf> leaves = route_leaves(inst, route.data(), m);
    RouteTree bbt;
    LcaTree lca;
    BenchResult result;
    result.evals = num_updates;

    {
        const auto t0 = Clock::now();
        if (method == 1) {
            bbt.build(leaves);
        } else {
            lca.build(leaves);
        }
        result.build_ns_total =
            std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
    }

    const auto t0 = Clock::now();
    for (std::int64_t t = 0; t < num_updates; ++t) {
        const std::int64_t pos = pick_pos(rng);
        std::int32_t c = pick_cust(rng);
        while (c == route[static_cast<std::size_t>(pos - 1)]) c = pick_cust(rng);
        // Replace leaf `pos` by the bridge for customer c at that position —
        // the state change an accepted relocate/exchange produces.
        Pwlf fn = detail::bridge_leaf(
            inst, route[static_cast<std::size_t>(pos - 1)], c);
        Pwlf full;
        if (method == 1) {
            leaves[static_cast<std::size_t>(pos)] = std::move(fn);
            bbt.build(leaves);  // Visser: full rebuild on change
            full = bbt.root();
        } else {
            lca.update_leaf(pos, std::move(fn));  // Blauth: localized update
            full = lca.query(0, m);
        }
        if (!full.xs.empty()) {
            const kayros::MinShift s = kayros::min_shifted_image(kayros::view(full));
            result.feasible += 1;
            result.checksum += s.value;
        }
    }
    const double total_ns =
        std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
    result.ns_per_eval = total_ns / static_cast<double>(result.evals);
    return result;
}

BenchResult bench_insertion_scan(const Instance& inst,
                                 const std::vector<std::int32_t>& route,
                                 std::int64_t num_candidates,
                                 std::uint64_t seed, int method) {
    using kayros::Pwlf;
    using kayros::compose;
    using kayros::identity;
    using kayros::view;
    const std::int64_t m = static_cast<std::int64_t>(route.size());
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::int32_t> pick_cust(1, inst.num_customers);
    std::vector<std::int32_t> candidates;
    candidates.reserve(static_cast<std::size_t>(num_candidates));
    for (std::int64_t k = 0; k < num_candidates; ++k) {
        candidates.push_back(pick_cust(rng));
    }

    LcaTree tree;
    BenchResult result;
    result.evals = num_candidates * (m + 1);
    {
        const auto t0 = Clock::now();
        tree.build(route_leaves(inst, route.data(), m));
        result.build_ns_total =
            std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
    }

    double dep_lo = inst.horizon_start;
    double dep_hi = inst.horizon_end;
    if (inst.has_time_windows) {
        dep_lo = std::max(dep_lo, inst.tw_earliest[0]);
        dep_hi = std::min(dep_hi, inst.tw_latest[0]);
    }
    const Pwlf dep = identity(dep_lo, dep_hi);

    auto score_from = [&](const Pwlf& prefix, std::int64_t i, std::int32_t c) {
        // prefix covers service completion at route[i-1] (or the depot
        // departure window for i == 0); insert c before route[i].
        const std::int32_t before =
            i > 0 ? route[static_cast<std::size_t>(i - 1)] : 0;
        if (c == before || (i < m && c == route[static_cast<std::size_t>(i)])) {
            return;  // self arc: no such move
        }
        Pwlf acc = compose(view(detail::bridge_leaf(inst, before, c)), view(prefix));
        if (acc.xs.empty()) return;
        if (i < m) {
            acc = compose(view(detail::bridge_leaf(
                              inst, c, route[static_cast<std::size_t>(i)])),
                          view(acc));
            if (acc.xs.empty()) return;
            if (i + 1 <= m) {
                const Pwlf suffix = tree.query(i + 1, m);
                if (suffix.xs.empty()) return;
                acc = compose(view(suffix), view(acc));
            }
        } else {
            acc = compose(view(detail::return_leaf(inst, c)), view(acc));
        }
        if (acc.xs.empty()) return;
        const kayros::MinShift s = kayros::min_shifted_image(view(acc));
        result.feasible += 1;
        result.checksum += s.value;
    };

    const auto t0 = Clock::now();
    if (method == 0) {
        // Per-position tree query for the prefix, every candidate.
        for (const std::int32_t c : candidates) {
            for (std::int64_t i = 0; i <= m; ++i) {
                const Pwlf prefix = i == 0 ? dep : tree.query(0, i - 1);
                if (prefix.xs.empty()) break;
                score_from(prefix, i, c);
            }
        }
    } else {
        // Incremental left-fold prefix, advanced once per position and
        // shared across all candidates at that position.
        Pwlf prefix = dep;
        for (std::int64_t i = 0; i <= m; ++i) {
            if (i > 0) {
                prefix = compose(view(tree.leaf(i - 1)), view(prefix));
                if (prefix.xs.empty()) break;
            }
            for (const std::int32_t c : candidates) score_from(prefix, i, c);
        }
    }
    const double total_ns =
        std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
    result.ns_per_eval = total_ns / static_cast<double>(result.evals);
    return result;
}

}  // namespace trt
