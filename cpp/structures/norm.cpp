#include "structures/norm.h"

#include <chrono>
#include <cmath>

#include "structures/baseline.h"

namespace trt {

using kayros::Instance;
using kayros::MinShift;
using kayros::Pwlf;
using kayros::compose;
using kayros::min_shifted_image;
using kayros::view;

namespace {

// goc EPS from pwlf_compare src/main.h (constexpr double EPS = 10e-7).
constexpr double kGocEps = 10e-7;

}  // namespace

void prune_inplace(Pwlf& f, NormMode mode) {
    if (mode == NormMode::none) return;
    if (mode == NormMode::safe_int) mode = NormMode::safe;
    const std::size_t n = f.xs.size();
    if (n < 3) return;
    std::vector<double> xs, ys;
    xs.reserve(n);
    ys.reserve(n);
    auto push = [&](double x, double y) {
        while (xs.size() >= 2) {
            const std::size_t k = xs.size();
            const double ax = xs[k - 2], ay = ys[k - 2];
            const double bx = xs[k - 1], by = ys[k - 1];
            bool drop = false;
            if (mode == NormMode::safe || mode == NormMode::collinear) {
                drop = (ay == by && by == y) || (ax == bx && bx == x);
                if (!drop && mode == NormMode::collinear && ax < bx && bx < x) {
                    // Exact evaluate() arithmetic on the long chord.
                    const double t = (bx - ax) / (x - ax);
                    drop = (ay + t * (y - ay)) == by;
                }
            } else {  // eps_slope: pwlf_compare normalized_add, verbatim logic
                const double s1 = (by - ay) / (bx - ax);
                const double s2 = (y - by) / (x - bx);
                if (std::fabs(s1 - s2) < kGocEps) {
                    drop = true;
                    if (!(std::fabs(s1 - 0.0) >= kGocEps)) y = by;
                }
            }
            if (!drop) break;
            xs.pop_back();
            ys.pop_back();
        }
        xs.push_back(x);
        ys.push_back(y);
    };
    push(f.xs[0], f.ys[0]);
    for (std::size_t k = 1; k < n; ++k) push(f.xs[k], f.ys[k]);
    f.xs = std::move(xs);
    f.ys = std::move(ys);
}

// The frozen kayros::compose event loop (cpp/pwlf/pwlf.cpp) with the safe
// flat/vertical dedup folded into emit — extends the frozen emit's exact
// duplicate check to whole runs, dropping the interior point in place.
Pwlf compose_safe_integrated(kayros::PwlfView f, kayros::PwlfView g) {
    if (f.n == 0 || g.n == 0) return {};
    const double lo = std::max(f.xs[0], g.ys[0]);
    const double hi = std::min(f.xs[f.n - 1], g.ys[g.n - 1]);
    if (lo > hi) return {};

    const double* fx = f.xs;
    const double* fy = f.ys;
    const double* gx = g.xs;
    const double* gy = g.ys;
    const std::int64_t nf = f.n, ng = g.n;

    std::int64_t i = 0;
    while (fx[i] < lo) ++i;
    std::int64_t j = 0;
    while (gy[j] < lo) ++j;

    Pwlf h;
    h.xs.reserve(static_cast<std::size_t>(nf + ng));
    h.ys.reserve(static_cast<std::size_t>(nf + ng));
    auto emit = [&h](double x, double y) {
        if (!h.xs.empty()) {
            if (x < h.xs.back()) x = h.xs.back();
            if (y < h.ys.back()) y = h.ys.back();
            if (x == h.xs.back() && y == h.ys.back()) return;
            const std::size_t k = h.xs.size();
            if (k >= 2 &&
                ((h.ys[k - 2] == h.ys[k - 1] && h.ys[k - 1] == y) ||
                 (h.xs[k - 2] == h.xs[k - 1] && h.xs[k - 1] == x))) {
                h.xs.back() = x;  // extend the flat/vertical run in place
                h.ys.back() = y;
                return;
            }
        }
        h.xs.push_back(x);
        h.ys.push_back(y);
    };

    std::vector<double> f_ys;
    std::vector<double> g_xs;
    while (true) {
        const bool has_f = i < nf && fx[i] <= hi;
        const bool has_g = j < ng && gy[j] <= hi;
        if (!has_f && !has_g) break;
        double u;
        if (!has_g) {
            u = fx[i];
        } else if (!has_f) {
            u = gy[j];
        } else {
            u = fx[i] <= gy[j] ? fx[i] : gy[j];
        }

        f_ys.clear();
        g_xs.clear();
        while (i < nf && fx[i] == u) {
            f_ys.push_back(fy[i]);
            ++i;
        }
        while (j < ng && gy[j] == u) {
            g_xs.push_back(gx[j]);
            ++j;
        }

        if (f_ys.empty()) {
            const double x_lo = fx[i - 1], x_hi = fx[i];
            const double t = (u - x_lo) / (x_hi - x_lo);
            f_ys.push_back(fy[i - 1] + t * (fy[i] - fy[i - 1]));
        }
        if (g_xs.empty()) {
            const double y_lo = gy[j - 1], y_hi = gy[j];
            const double t = (u - y_lo) / (y_hi - y_lo);
            g_xs.push_back(gx[j - 1] + t * (gx[j] - gx[j - 1]));
        }

        for (double x_val : g_xs) emit(x_val, f_ys[0]);
        for (std::size_t k = 1; k < f_ys.size(); ++k) emit(g_xs.back(), f_ys[k]);
    }

    return h;
}

std::vector<Pwlf> route_leaves_norm(const Instance& inst,
                                    const std::int32_t* route, std::int64_t len,
                                    NormMode mode) {
    std::vector<Pwlf> leaves = route_leaves(inst, route, len);
    if (mode != NormMode::none) {
        for (Pwlf& leaf : leaves) prune_inplace(leaf, mode);
    }
    return leaves;
}

Pwlf fold_left_norm(const std::vector<Pwlf>& leaves, NormMode mode) {
    if (leaves.empty()) return {};
    Pwlf acc = leaves.front();
    for (std::size_t i = 1; i < leaves.size(); ++i) {
        if (acc.xs.empty()) return acc;
        acc = compose_norm(view(leaves[i]), view(acc), mode);
    }
    return acc;
}

FoldProfile fold_profile(const Instance& inst, const std::int32_t* route,
                         std::int64_t len, NormMode mode) {
    FoldProfile out;
    const std::vector<Pwlf> leaves = route_leaves_norm(inst, route, len, mode);
    if (leaves.empty()) return out;
    out.prefix_bp.reserve(leaves.size());
    Pwlf acc = leaves.front();
    out.prefix_bp.push_back(static_cast<std::int64_t>(acc.xs.size()));
    for (std::size_t i = 1; i < leaves.size(); ++i) {
        if (acc.xs.empty()) break;
        acc = compose_norm(view(leaves[i]), view(acc), mode);
        out.prefix_bp.push_back(static_cast<std::int64_t>(acc.xs.size()));
    }
    if (!acc.xs.empty()) {
        const MinShift s = min_shifted_image(view(acc));
        out.feasible = true;
        out.delta_star = s.value;
        out.argmin_x = s.argmin_x;
    }
    out.delta = std::move(acc);
    return out;
}

FoldBench bench_fold(const Instance& inst,
                     const std::vector<std::vector<std::int32_t>>& routes,
                     std::int64_t repeats, NormMode mode) {
    using Clock = std::chrono::steady_clock;
    FoldBench result;
    const auto t0 = Clock::now();
    for (std::int64_t rep = 0; rep < repeats; ++rep) {
        for (const auto& route : routes) {
            const std::vector<Pwlf> leaves = route_leaves_norm(
                inst, route.data(), static_cast<std::int64_t>(route.size()),
                mode);
            const Pwlf delta = fold_left_norm(leaves, mode);
            result.folds += 1;
            if (!delta.xs.empty()) {
                const MinShift s = min_shifted_image(view(delta));
                result.feasible += 1;
                result.checksum += s.value;
            }
        }
    }
    const double total_ns =
        std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
    result.ns_per_fold = total_ns / static_cast<double>(result.folds);
    return result;
}

}  // namespace trt
