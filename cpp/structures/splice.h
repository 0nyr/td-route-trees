#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "core/instance.h"
#include "structures/baseline.h"

namespace trt {

namespace detail {

inline kayros::Pwlf bridge_leaf(const kayros::Instance& inst, std::int32_t from,
                                std::int32_t to) {
    const kayros::PwlfView alpha = inst.arc(from, to);
    if (alpha.n == 0) return {};
    const double service_time = inst.service_times[to];
    kayros::Pwlf theta;
    if (inst.has_time_windows) {
        theta = kayros::make_theta(inst.tw_earliest[to], inst.tw_latest[to],
                                   service_time);
    } else {
        theta = kayros::Pwlf{{0.0, alpha.ys[alpha.n - 1]},
                             {service_time, alpha.ys[alpha.n - 1] + service_time}};
    }
    return kayros::compose(kayros::view(theta), alpha);
}

inline kayros::Pwlf return_leaf(const kayros::Instance& inst, std::int32_t from) {
    const kayros::PwlfView alpha = inst.arc(from, 0);
    kayros::Pwlf tail{std::vector<double>(alpha.xs, alpha.xs + alpha.n),
                      std::vector<double>(alpha.ys, alpha.ys + alpha.n)};
    if (inst.has_time_windows && !tail.xs.empty()) {
        const kayros::Pwlf clamp = kayros::identity(0.0, inst.tw_latest[0]);
        tail = kayros::compose(kayros::view(clamp), kayros::view(tail));
    }
    return tail;
}

}  // namespace detail

// Exchange-move evaluation on any structure exposing query(lo, hi) over the
// P4.0 leaf convention: the new first route r1' = r1[0..i1-1] ++ r2[i2..j2]
// ++ r1[j1+1..], with empty segments encoded as j < i. Insertion, deletion,
// relocate, swap and 2-opt* are special cases.
template <class Tree1, class Tree2>
kayros::RouteEval eval_spliced_impl(const kayros::Instance& inst,
                                    const Tree1& tree1,
                                    const std::vector<std::int32_t>& route1,
                                    std::int64_t i1, std::int64_t j1,
                                    const Tree2& tree2,
                                    const std::vector<std::int32_t>& route2,
                                    std::int64_t i2, std::int64_t j2) {
    using kayros::Pwlf;
    using kayros::compose;
    using kayros::identity;
    using kayros::view;

    const std::int64_t m1 = static_cast<std::int64_t>(route1.size());
    const std::int64_t m2 = static_cast<std::int64_t>(route2.size());
    const bool incoming = i2 <= j2;
    if (i1 < 0 || i1 > m1 || j1 < i1 - 1 || j1 >= m1) {
        throw std::invalid_argument("invalid [i1, j1]");
    }
    if (incoming && (i2 < 0 || j2 >= m2)) {
        throw std::invalid_argument("invalid [i2, j2]");
    }
    const bool head = i1 > 0;
    const bool tail = j1 + 1 < m1;
    if (!head && !incoming && !tail) return {false, 0.0, 0.0};

    Pwlf acc;
    if (head) {
        acc = tree1.query(0, i1 - 1);
    } else {
        double dep_lo = inst.horizon_start;
        double dep_hi = inst.horizon_end;
        if (inst.has_time_windows) {
            dep_lo = std::max(dep_lo, inst.tw_earliest[0]);
            dep_hi = std::min(dep_hi, inst.tw_latest[0]);
        }
        if (dep_lo > dep_hi) return {false, 0.0, 0.0};
        acc = identity(dep_lo, dep_hi);
    }
    if (acc.xs.empty()) return {false, 0.0, 0.0};

    std::int32_t last = head ? route1[static_cast<std::size_t>(i1 - 1)] : 0;

    if (incoming) {
        const Pwlf bridge =
            detail::bridge_leaf(inst, last, route2[static_cast<std::size_t>(i2)]);
        if (bridge.xs.empty()) return {false, 0.0, 0.0};
        acc = compose(view(bridge), view(acc));
        if (acc.xs.empty()) return {false, 0.0, 0.0};
        if (j2 > i2) {
            const Pwlf middle = tree2.query(i2 + 1, j2);
            if (middle.xs.empty()) return {false, 0.0, 0.0};
            acc = compose(view(middle), view(acc));
            if (acc.xs.empty()) return {false, 0.0, 0.0};
        }
        last = route2[static_cast<std::size_t>(j2)];
    }

    if (tail) {
        const Pwlf bridge =
            detail::bridge_leaf(inst, last, route1[static_cast<std::size_t>(j1 + 1)]);
        if (bridge.xs.empty()) return {false, 0.0, 0.0};
        acc = compose(view(bridge), view(acc));
        if (acc.xs.empty()) return {false, 0.0, 0.0};
        if (j1 + 2 <= m1) {
            const Pwlf suffix = tree1.query(j1 + 2, m1);
            if (suffix.xs.empty()) return {false, 0.0, 0.0};
            acc = compose(view(suffix), view(acc));
        }
    } else {
        const Pwlf ret = detail::return_leaf(inst, last);
        if (ret.xs.empty()) return {false, 0.0, 0.0};
        acc = compose(view(ret), view(acc));
    }
    if (acc.xs.empty()) return {false, 0.0, 0.0};
    const kayros::MinShift s = kayros::min_shifted_image(view(acc));
    return {true, s.value, s.argmin_x};
}

}  // namespace trt
