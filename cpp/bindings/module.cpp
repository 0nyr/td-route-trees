#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "core/instance.h"
#include "structures/baseline.h"
#include "structures/bbt.h"
#include "structures/bench.h"
#include "structures/lca_bst.h"
#include "structures/norm.h"
#include "structures/splice.h"

namespace py = pybind11;

#ifndef TRT_VERSION
#define TRT_VERSION "0.0.0"
#endif

namespace {

using ArcSpec = std::tuple<std::int32_t, std::int32_t, std::vector<double>,
                           std::vector<double>>;

// Identical construction contract to kayros._core.Instance (frozen alongside
// the engine): flat ATF pool addressed by (i, j) with 0 = depot.
kayros::Instance make_instance(
    std::int32_t num_customers, std::optional<std::int32_t> num_vehicles,
    std::int64_t vehicle_capacity, std::pair<double, double> horizon,
    std::optional<std::vector<std::pair<double, double>>> time_windows,
    std::vector<std::int64_t> demands, std::vector<double> service_times,
    const std::vector<ArcSpec>& arcs) {
    kayros::Instance inst;
    inst.num_customers = num_customers;
    inst.num_vehicles = num_vehicles.value_or(-1);
    inst.vehicle_capacity = vehicle_capacity;
    inst.horizon_start = horizon.first;
    inst.horizon_end = horizon.second;
    const std::int64_t nv = inst.num_vertices();
    const std::size_t expected = static_cast<std::size_t>(nv);
    if (demands.size() != expected || service_times.size() != expected) {
        throw std::invalid_argument(
            "demands and service_times must have num_customers + 1 entries");
    }
    inst.demands = std::move(demands);
    inst.service_times = std::move(service_times);
    if (time_windows.has_value()) {
        if (time_windows->size() != expected) {
            throw std::invalid_argument(
                "time_windows must have num_customers + 1 entries");
        }
        inst.has_time_windows = true;
        inst.tw_earliest.reserve(expected);
        inst.tw_latest.reserve(expected);
        for (const auto& [earliest, latest] : *time_windows) {
            inst.tw_earliest.push_back(earliest);
            inst.tw_latest.push_back(latest);
        }
    }

    const std::int64_t num_arcs = nv * nv;
    std::vector<std::int64_t> lengths(static_cast<std::size_t>(num_arcs), 0);
    std::int64_t total = 0;
    for (const auto& [i, j, xs, ys] : arcs) {
        if (i < 0 || i >= nv || j < 0 || j >= nv || i == j) {
            throw std::invalid_argument("invalid arc endpoints");
        }
        if (xs.size() != ys.size() || xs.size() < 2) {
            throw std::invalid_argument("arc ATF must have >= 2 breakpoints");
        }
        const std::int64_t a = static_cast<std::int64_t>(i) * nv + j;
        if (lengths[static_cast<std::size_t>(a)] != 0) {
            throw std::invalid_argument("duplicate arc");
        }
        lengths[static_cast<std::size_t>(a)] =
            static_cast<std::int64_t>(xs.size());
        total += static_cast<std::int64_t>(xs.size());
    }
    inst.atf_offset.assign(static_cast<std::size_t>(num_arcs) + 1, 0);
    for (std::int64_t a = 0; a < num_arcs; ++a) {
        inst.atf_offset[static_cast<std::size_t>(a) + 1] =
            inst.atf_offset[static_cast<std::size_t>(a)] +
            lengths[static_cast<std::size_t>(a)];
    }
    inst.atf_xs.assign(static_cast<std::size_t>(total), 0.0);
    inst.atf_ys.assign(static_cast<std::size_t>(total), 0.0);
    for (const auto& [i, j, xs, ys] : arcs) {
        const std::int64_t a = static_cast<std::int64_t>(i) * nv + j;
        const std::int64_t b = inst.atf_offset[static_cast<std::size_t>(a)];
        std::copy(xs.begin(), xs.end(),
                  inst.atf_xs.begin() + static_cast<std::ptrdiff_t>(b));
        std::copy(ys.begin(), ys.end(),
                  inst.atf_ys.begin() + static_cast<std::ptrdiff_t>(b));
    }
    return inst;
}

std::vector<std::int32_t> checked_route(const kayros::Instance& inst,
                                        const std::vector<std::int32_t>& route) {
    if (route.empty()) throw std::invalid_argument("route must be non-empty");
    for (const std::int32_t v : route) {
        if (v < 1 || v > inst.num_customers) {
            throw std::invalid_argument("route vertex out of range");
        }
    }
    return route;
}

py::tuple pwlf_tuple(kayros::Pwlf f) {
    return py::make_tuple(std::move(f.xs), std::move(f.ys));
}

trt::NormMode to_mode(int mode) {
    if (mode < 0 || mode > 4) throw std::invalid_argument("mode must be 0..4");
    return static_cast<trt::NormMode>(mode);
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() =
        "td-route-trees compiled core: frozen NDCPWLF engine + TD move "
        "evaluation structures under study";
    m.attr("__version__") = TRT_VERSION;

    py::class_<kayros::Instance>(m, "Instance")
        .def(py::init(&make_instance), py::arg("num_customers"),
             py::arg("num_vehicles"), py::arg("vehicle_capacity"),
             py::arg("horizon"), py::arg("time_windows"), py::arg("demands"),
             py::arg("service_times"), py::arg("arcs"))
        .def_readonly("num_customers", &kayros::Instance::num_customers)
        .def_readonly("has_time_windows", &kayros::Instance::has_time_windows);

    // Checker-identical route pricing (frozen kayros fold).
    m.def("evaluate_route",
          [](const kayros::Instance& inst, const std::vector<std::int32_t>& route) {
              const auto r = checked_route(inst, route);
              const kayros::RouteEval e =
                  kayros::evaluate_route(inst, r.data(),
                                         static_cast<std::int64_t>(r.size()));
              return py::make_tuple(e.feasible, e.duration, e.departure);
          });

    // delta_r via the checker's own association (alpha then theta, left fold).
    m.def("route_delta_checker",
          [](const kayros::Instance& inst, const std::vector<std::int32_t>& route) {
              const auto r = checked_route(inst, route);
              return pwlf_tuple(kayros::route_ready_time_function(
                  inst, r.data(), static_cast<std::int64_t>(r.size())));
          });

    // delta_r via precomposed leaves, left fold and balanced fold.
    m.def("route_delta_leaf_left",
          [](const kayros::Instance& inst, const std::vector<std::int32_t>& route) {
              const auto r = checked_route(inst, route);
              const auto leaves = trt::route_leaves(
                  inst, r.data(), static_cast<std::int64_t>(r.size()));
              return pwlf_tuple(trt::fold_left(leaves));
          });
    m.def("route_delta_balanced",
          [](const kayros::Instance& inst, const std::vector<std::int32_t>& route) {
              const auto r = checked_route(inst, route);
              const auto leaves = trt::route_leaves(
                  inst, r.data(), static_cast<std::int64_t>(r.size()));
              return pwlf_tuple(trt::fold_balanced(leaves));
          });

    m.def("min_shifted_image",
          [](std::vector<double> xs, std::vector<double> ys) {
              const kayros::Pwlf f{std::move(xs), std::move(ys)};
              if (f.xs.empty()) throw std::invalid_argument("empty function");
              const kayros::MinShift s = kayros::min_shifted_image(kayros::view(f));
              return py::make_tuple(s.value, s.argmin_x);
          });

    // --- P4.2: Visser ready-time-function tree ---
    py::class_<trt::RouteTree>(m, "RouteTree")
        .def(py::init([](const kayros::Instance& inst,
                         const std::vector<std::int32_t>& route, int mode) {
                 const auto r = checked_route(inst, route);
                 trt::RouteTree tree;
                 tree.build(trt::route_leaves_norm(
                                inst, r.data(),
                                static_cast<std::int64_t>(r.size()),
                                to_mode(mode)),
                            to_mode(mode));
                 return tree;
             }),
             py::arg("instance"), py::arg("route"), py::arg("mode") = 0)
        .def_property_readonly("num_leaves", &trt::RouteTree::num_leaves)
        .def_property_readonly("total_stored_bp", &trt::RouteTree::total_stored_bp)
        .def("root",
             [](const trt::RouteTree& tree) {
                 kayros::Pwlf f = tree.root();
                 return pwlf_tuple(std::move(f));
             })
        .def("query", [](const trt::RouteTree& tree, std::int64_t lo,
                         std::int64_t hi) { return pwlf_tuple(tree.query(lo, hi)); });

    m.def("eval_spliced",
          [](const kayros::Instance& inst, const trt::RouteTree& tree1,
             const std::vector<std::int32_t>& route1, std::int64_t i1,
             std::int64_t j1, const trt::RouteTree& tree2,
             const std::vector<std::int32_t>& route2, std::int64_t i2,
             std::int64_t j2) {
              const kayros::RouteEval e = trt::eval_spliced(
                  inst, tree1, route1, i1, j1, tree2, route2, i2, j2);
              return py::make_tuple(e.feasible, e.duration, e.departure);
          });

    // --- P4.3: Blauth LCA-BST ---
    py::class_<trt::LcaTree>(m, "LcaTree")
        .def(py::init([](const kayros::Instance& inst,
                         const std::vector<std::int32_t>& route, int mode) {
                 const auto r = checked_route(inst, route);
                 trt::LcaTree tree;
                 tree.build(trt::route_leaves_norm(
                                inst, r.data(),
                                static_cast<std::int64_t>(r.size()),
                                to_mode(mode)),
                            to_mode(mode));
                 return tree;
             }),
             py::arg("instance"), py::arg("route"), py::arg("mode") = 0)
        .def_property_readonly("num_leaves", &trt::LcaTree::num_leaves)
        .def_property_readonly("total_stored_bp", &trt::LcaTree::total_stored_bp)
        .def("query", [](const trt::LcaTree& tree, std::int64_t lo,
                         std::int64_t hi) { return pwlf_tuple(tree.query(lo, hi)); })
        .def("update_leaf",
             [](trt::LcaTree& tree, std::int64_t leaf, std::vector<double> xs,
                std::vector<double> ys) {
                 tree.update_leaf(leaf, kayros::Pwlf{std::move(xs), std::move(ys)});
             });

    m.def("eval_spliced",
          [](const kayros::Instance& inst, const trt::LcaTree& tree1,
             const std::vector<std::int32_t>& route1, std::int64_t i1,
             std::int64_t j1, const trt::LcaTree& tree2,
             const std::vector<std::int32_t>& route2, std::int64_t i2,
             std::int64_t j2) {
              const kayros::RouteEval e = trt::eval_spliced_impl(
                  inst, tree1, route1, i1, j1, tree2, route2, i2, j2);
              return py::make_tuple(e.feasible, e.duration, e.departure);
          });

    m.def("bench_exchange",
          [](const kayros::Instance& inst,
             const std::vector<std::vector<std::int32_t>>& routes,
             std::int64_t num_moves, std::int64_t max_seg, std::uint64_t seed,
             int method, int mode) {
              const trt::BenchResult r = trt::bench_exchange(
                  inst, routes, num_moves, max_seg, seed, method, to_mode(mode));
              py::dict out;
              out["ns_per_eval"] = r.ns_per_eval;
              out["evals"] = r.evals;
              out["feasible"] = r.feasible;
              out["checksum"] = r.checksum;
              out["build_ns_total"] = r.build_ns_total;
              return out;
          },
          py::arg("instance"), py::arg("routes"), py::arg("num_moves"),
          py::arg("max_seg"), py::arg("seed"), py::arg("method"),
          py::arg("mode") = 0);

    m.def("bench_update",
          [](const kayros::Instance& inst, const std::vector<std::int32_t>& route,
             std::int64_t num_updates, std::uint64_t seed, int method) {
              const trt::BenchResult r =
                  trt::bench_update(inst, route, num_updates, seed, method);
              py::dict out;
              out["ns_per_eval"] = r.ns_per_eval;
              out["evals"] = r.evals;
              out["feasible"] = r.feasible;
              out["checksum"] = r.checksum;
              out["build_ns_total"] = r.build_ns_total;
              return out;
          },
          py::arg("instance"), py::arg("route"), py::arg("num_updates"),
          py::arg("seed"), py::arg("method"));

    m.def("bench_insertion_scan",
          [](const kayros::Instance& inst, const std::vector<std::int32_t>& route,
             std::int64_t num_candidates, std::uint64_t seed, int method) {
              const trt::BenchResult r = trt::bench_insertion_scan(
                  inst, route, num_candidates, seed, method);
              py::dict out;
              out["ns_per_eval"] = r.ns_per_eval;
              out["evals"] = r.evals;
              out["feasible"] = r.feasible;
              out["checksum"] = r.checksum;
              out["build_ns_total"] = r.build_ns_total;
              return out;
          },
          py::arg("instance"), py::arg("route"), py::arg("num_candidates"),
          py::arg("seed"), py::arg("method"));

    // --- Normalization study (norm-vs-no-norm checker question) ---
    m.def("prune",
          [](std::vector<double> xs, std::vector<double> ys, int mode) {
              kayros::Pwlf f{std::move(xs), std::move(ys)};
              trt::prune_inplace(f, to_mode(mode));
              return pwlf_tuple(std::move(f));
          },
          py::arg("xs"), py::arg("ys"), py::arg("mode"));

    m.def("evaluate_pwlf",
          [](std::vector<double> xs, std::vector<double> ys, double x) {
              const kayros::Pwlf f{std::move(xs), std::move(ys)};
              return kayros::evaluate(kayros::view(f), x);
          },
          py::arg("xs"), py::arg("ys"), py::arg("x"));

    m.def("fold_profile",
          [](const kayros::Instance& inst,
             const std::vector<std::int32_t>& route, int mode) {
              const auto r = checked_route(inst, route);
              trt::FoldProfile p = trt::fold_profile(
                  inst, r.data(), static_cast<std::int64_t>(r.size()),
                  to_mode(mode));
              py::dict out;
              out["feasible"] = p.feasible;
              out["delta_star"] = p.delta_star;
              out["argmin_x"] = p.argmin_x;
              out["prefix_bp"] = p.prefix_bp;
              out["xs"] = std::move(p.delta.xs);
              out["ys"] = std::move(p.delta.ys);
              return out;
          },
          py::arg("instance"), py::arg("route"), py::arg("mode") = 0);

    m.def("bench_fold",
          [](const kayros::Instance& inst,
             const std::vector<std::vector<std::int32_t>>& routes,
             std::int64_t repeats, int mode) {
              const trt::FoldBench b =
                  trt::bench_fold(inst, routes, repeats, to_mode(mode));
              py::dict out;
              out["ns_per_fold"] = b.ns_per_fold;
              out["folds"] = b.folds;
              out["feasible"] = b.feasible;
              out["checksum"] = b.checksum;
              return out;
          },
          py::arg("instance"), py::arg("routes"), py::arg("repeats"),
          py::arg("mode") = 0);
}
