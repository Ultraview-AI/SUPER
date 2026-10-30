// Benchmark of the A* clearance cost on a real hangar / aircraft map (not run by colcon test).
//   bench_astar_clearance <map.pcd> <n_queries> <path_dump_prefix> [w:d ...]
// Uses the Aegis front-end settings (ROG-Map 0.2 m, inflation 0.2 m x 2, A* box 500x500x100, horizon
// 7 m, inflated-map search). Start/goal pairs are random free cells 0.8-3 m from the cloud and 3-12 m
// apart (seeded, so a stock build replays the same queries). Per weight config it prints success,
// time per call, expansions, path length and raw-point clearance, and dumps every path so the w = 0
// output can be diffed against a build of the unmodified planner (-DASTAR_STOCK).
#include "astar_test_utils.hpp"

#include <rclcpp/rclcpp.hpp>

#include <chrono>
#include <cstdio>
#include <random>

using namespace astar_test;
using rog_map::Vec3f;

namespace {
    struct Cfg {
        double w, d;
    };

    double pct(std::vector<double> v, double q) {
        if (v.empty()) {
            return 0;
        }
        std::sort(v.begin(), v.end());
        return v[std::min(v.size() - 1, static_cast<size_t>(q * (v.size() - 1) + 0.5))];
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <map.pcd> <n_queries> <dump_prefix> [w:d ...]\n", argv[0]);
        return 2;
    }
    rclcpp::init(argc, argv);
    const std::string pcd = argv[1];
    const int n_queries = std::stoi(argv[2]);
    const std::string dump = argv[3];
    std::vector<Cfg> cfgs;
    for (int i = 4; i < argc; i++) {
        double w, d;
        if (std::sscanf(argv[i], "%lf:%lf", &w, &d) == 2) {
            cfgs.push_back({w, d});
        }
    }
    if (cfgs.empty()) {
        cfgs = {{0, 0}};
    }
#ifdef ASTAR_STOCK
    cfgs = {{0, 0}};
#endif

    rog_map::PointCloud::Ptr cloud(new rog_map::PointCloud);
    if (pcl::io::loadPCDFile(pcd, *cloud) != 0) {
        std::fprintf(stderr, "cannot load %s\n", pcd.c_str());
        return 1;
    }
    Eigen::Vector3d lo(1e9, 1e9, 1e9), hi(-1e9, -1e9, -1e9);
    for (const auto &p: cloud->points) {
        lo = lo.cwiseMin(Eigen::Vector3d(p.x, p.y, p.z));
        hi = hi.cwiseMax(Eigen::Vector3d(p.x, p.y, p.z));
    }
    std::printf("cloud %zu pts, bbox [%.1f %.1f %.1f] - [%.1f %.1f %.1f]\n", cloud->size(), lo.x(), lo.y(),
                lo.z(), hi.x(), hi.y(), hi.z());

    MapSpec spec;  // Aegis aegis_super.yaml rog_map / astar values
    spec.resolution = 0.2;
    spec.inflation_resolution = 0.2;
    spec.inflation_step = 2;
    spec.map_size = Eigen::Vector3d(60, 60, 16);
    const Eigen::Vector3d c = (lo + hi) / 2;
    spec.origin = Eigen::Vector3d(c.x(), c.y(), 6.0);
    spec.virtual_ground = -2.0;
    spec.virtual_ceil = 14.4;
    spec.astar_vox = Eigen::Vector3i(500, 500, 100);
    const std::string cfg_path = writeConfig(spec, pcd, "bench");

    auto node = std::make_shared<rclcpp::Node>("bench_astar_clearance");
    Planner p = makePlanner(node, cfg_path);
    pcl::KdTreeFLANN<pcl::PointXYZI> kd;
    kd.setInputCloud(cloud);

    // Query sampling (seeded).
    std::mt19937 rng(20261029);
    const Eigen::Vector3d qlo(c.x() - 25, c.y() - 25, 0.8), qhi(c.x() + 25, c.y() + 25, 8.0);
    std::uniform_real_distribution<double> ux(qlo.x(), qhi.x()), uy(qlo.y(), qhi.y()), uz(qlo.z(), qhi.z());
    auto goodPoint = [&](const Vec3f &v) {
        if (p.map->getInfGridType(v) == rog_map::OCCUPIED) {
            return false;
        }
        const double d = pathClearance(rog_map::vec_Vec3f{v}, kd).min;
        return d > 0.8 && d < 3.0;
    };
    std::vector<std::pair<Vec3f, Vec3f>> queries;
    while (static_cast<int>(queries.size()) < n_queries) {
        Vec3f s(ux(rng), uy(rng), uz(rng));
        if (!goodPoint(s)) {
            continue;
        }
        for (int t = 0; t < 200; t++) {
            Vec3f g(ux(rng), uy(rng), uz(rng));
            const double dist = (g - s).norm();
            if (dist < 3.0 || dist > 12.0 || !goodPoint(g)) {
                continue;
            }
            queries.emplace_back(s, g);
            break;
        }
    }

    const double horizon = 7.0;  // super_planner/planning_horizon
    for (const auto &cf: cfgs) {
#ifndef ASTAR_STOCK
        p.astar->setClearanceCost(cf.w, cf.d);
#endif
        char name[128];
        std::snprintf(name, sizeof(name), "%s_w%.2f_d%.2f.txt", dump.c_str(), cf.w, cf.d);
        FILE *out = std::fopen(name, "w");
        std::vector<double> t_ms, exp, probes, len, cmin, cmed;
        int ok = 0, over_100ms = 0;
        // Warm-up (first call touches the node buffer pages).
        {
            rog_map::vec_Vec3f tmp;
            p.astar->pointToPointPathSearch(queries[0].first, queries[0].second, frontendFlags(), horizon, tmp, 5.0);
        }
        for (size_t i = 0; i < queries.size(); i++) {
            rog_map::vec_Vec3f path;
            const auto t0 = std::chrono::steady_clock::now();
            const auto ret = p.astar->pointToPointPathSearch(queries[i].first, queries[i].second, frontendFlags(),
                                                             horizon, path, 5.0);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(out, "q %zu ret %d n %zu\n", i, static_cast<int>(ret), path.size());
            for (const auto &v: path) {
                std::fprintf(out, "%.6f %.6f %.6f\n", v.x(), v.y(), v.z());
            }
            t_ms.push_back(ms);
            over_100ms += ms > 100.0;
#ifndef ASTAR_STOCK
            const auto st = p.astar->getLastSearchStats();
            exp.push_back(st.expanded);
            probes.push_back(static_cast<double>(st.clearance_probes));
#endif
            if (ret == path_search::REACH_GOAL || ret == path_search::REACH_HORIZON) {
                ok++;
                len.push_back(pathLength(path));
                // Skip the first and last 3 points (start / goal are sampled 0.8-3 m from the cloud).
                rog_map::vec_Vec3f mid;
                for (size_t k = 3; k + 3 < path.size(); k++) {
                    mid.push_back(path[k]);
                }
                if (!mid.empty()) {
                    const Clearance cl = pathClearance(mid, kd);
                    cmin.push_back(cl.min);
                    cmed.push_back(cl.median);
                }
            }
        }
        std::fclose(out);
        std::printf("w=%.2f d=%.2f: ok %d/%zu | ms p50 %.2f p95 %.2f max %.2f (>100ms: %d) | exp p50 %.0f p95 %.0f"
                    " | probes p50 %.0f p95 %.0f | len p50 %.2f | path min-clr p10 %.2f p50 %.2f | median-clr p50 %.2f\n",
                    cf.w, cf.d, ok, queries.size(), pct(t_ms, .5), pct(t_ms, .95), pct(t_ms, 1.0), over_100ms,
                    pct(exp, .5), pct(exp, .95), pct(probes, .5), pct(probes, .95), pct(len, .5), pct(cmin, .1),
                    pct(cmin, .5), pct(cmed, .5));
        std::fflush(stdout);
    }
    rclcpp::shutdown();
    return 0;
}
