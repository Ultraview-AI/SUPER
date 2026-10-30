// Unit tests for the A* obstacle-clearance cost (astar/clearance_weight, astar/clearance_dist).
#include "astar_test_utils.hpp"

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

using namespace astar_test;
using rog_map::Vec3f;

namespace {

    // A planar sheet of points, 0.1 m spacing. `keep` decides which points stay (for gaps).
    template<class Keep>
    void addSheet(rog_map::PointCloud &pc, const Vec3f &o, const Vec3f &u, const Vec3f &v,
                  double lu, double lv, Keep keep) {
        const double step = 0.1;
        for (double a = 0; a <= lu + 1e-9; a += step) {
            for (double b = 0; b <= lv + 1e-9; b += step) {
                const Vec3f p = o + a * u + b * v;
                if (!keep(p)) {
                    continue;
                }
                pcl::PointXYZI q;
                q.x = p.x();
                q.y = p.y();
                q.z = p.z();
                q.intensity = 100;
                pc.push_back(q);
            }
        }
    }

    // ROG-Map can only be initialised once per process, so all scenes share one map, side by side
    // along x (each A* query box only reaches its own scene):
    //   x ~ -30: a wall in the plane y = 0 (fly along a surface);
    //   x =   0: a wall in the plane x = 0 with a 3 m slot y in (-1, 2), the only way through;
    //   x =  30: the same with a 1.4 m slot y in (-0.7, 0.7), a few inflated-free cells wide.
    // Every wall spans the full map height; the slot walls span the full map width.
    struct World {
        rog_map::PointCloud cloud;
        std::string pcd;
        pcl::KdTreeFLANN<pcl::PointXYZI> kd;
        rclcpp::Node::SharedPtr node;
        Planner stock;       // config without the clearance keys
        Planner clr;         // clearance_weight 0 / clearance_dist 1.0 at start; tests switch it
    };
    World *W = nullptr;

    void buildWorld() {
        W = new World;
        auto all = [](const Vec3f &) { return true; };
        addSheet(W->cloud, Vec3f(-42, 0, -3), Vec3f::UnitX(), Vec3f::UnitZ(), 24, 10, all);
        addSheet(W->cloud, Vec3f(0, -15, -3), Vec3f::UnitY(), Vec3f::UnitZ(), 30, 10,
                 [](const Vec3f &p) { return p.y() <= -1.0 || p.y() >= 2.0; });
        addSheet(W->cloud, Vec3f(30, -15, -3), Vec3f::UnitY(), Vec3f::UnitZ(), 30, 10,
                 [](const Vec3f &p) { return p.y() <= -0.7 || p.y() >= 0.7; });
        W->cloud.width = W->cloud.size();
        W->cloud.height = 1;
        W->pcd = tmpPath("world.pcd");
        pcl::io::savePCDFileBinary(W->pcd, W->cloud);
        W->kd.setInputCloud(W->cloud.makeShared());

        MapSpec spec;
        spec.map_size = Eigen::Vector3d(90, 30, 8);
        W->node = std::make_shared<rclcpp::Node>("astar_clearance_test");
        spec.write_clearance_keys = false;
        const std::string cfg_stock = writeConfig(spec, W->pcd, "stock");
        W->stock = makePlanner(W->node, cfg_stock);
        spec.write_clearance_keys = true;
        spec.clearance_weight = 0.0;
        spec.clearance_dist = 1.0;
        const std::string cfg_clr = writeConfig(spec, W->pcd, "clr");
        W->clr = W->stock;
        W->clr.astar = std::make_shared<path_search::Astar>(cfg_clr, W->stock.ros, W->stock.map);
    }

    struct Result {
        path_search::RET_CODE ret;
        rog_map::vec_Vec3f path;
        path_search::Astar::SearchStats stats;
    };

    Result search(Planner &p, const Vec3f &s, const Vec3f &g, double horizon = 9999.0) {
        Result r;
        r.ret = p.astar->pointToPointPathSearch(s, g, frontendFlags(), horizon, r.path, 5.0);
        r.stats = p.astar->getLastSearchStats();
        return r;
    }

    // Path point closest to the plane x = 0 (where a path crosses the gap wall).
    Vec3f crossing(const rog_map::vec_Vec3f &path) {
        Vec3f best = path.front();
        for (const auto &p: path) {
            if (std::abs(p.x()) < std::abs(best.x())) {
                best = p;
            }
        }
        return best;
    }

    double clearanceAt(const Vec3f &p, pcl::KdTreeFLANN<pcl::PointXYZI> &kd) {
        return pathClearance(rog_map::vec_Vec3f{p}, kd).min;
    }
}

// Weight 0 is the stock search: no clearance query is made, and the path is identical to the one from
// a config without the clearance keys at all, and stays identical when only clearance_dist is set.
TEST(AstarClearance, ZeroWeightIsStock) {
    Planner &stock = W->stock;
    Planner &w0 = W->clr;
    w0.astar->setClearanceCost(0.0, 1.0);

    const std::vector<std::pair<Vec3f, Vec3f>> queries = {
            {Vec3f(-5, -3, 1.5), Vec3f(5, -3, 1.5)},
            {Vec3f(-4, 4, 0.5),  Vec3f(6, -6, 2.5)},
            {Vec3f(-2, -8, 3.0), Vec3f(3, 8, 0.0)},
            {Vec3f(-35, 0.9, 1.5), Vec3f(-25, 0.9, 1.5)},
            {Vec3f(25, -3, 1.5), Vec3f(35, 2, 1.5)},
    };
    for (const auto &q: queries) {
        for (double horizon: {9999.0, 4.0}) {
            Result a = search(stock, q.first, q.second, horizon);
            Result b = search(w0, q.first, q.second, horizon);
            ASSERT_EQ(a.ret, b.ret);
            ASSERT_EQ(a.path.size(), b.path.size());
            for (size_t i = 0; i < a.path.size(); i++) {
                EXPECT_EQ(a.path[i], b.path[i]) << "point " << i;
            }
            EXPECT_EQ(a.stats.expanded, b.stats.expanded);
            EXPECT_EQ(a.stats.clearance_probes, 0);
            EXPECT_EQ(b.stats.clearance_probes, 0);
        }
    }
    // Switching the cost on and back off at run time restores the stock path too.
    Result before = search(w0, queries[0].first, queries[0].second);
    w0.astar->setClearanceCost(1.0, 1.0);
    Result on = search(w0, queries[0].first, queries[0].second);
    EXPECT_GT(on.stats.clearance_probes, 0);
    w0.astar->setClearanceCost(0.0, 1.0);
    Result after = search(w0, queries[0].first, queries[0].second);
    EXPECT_EQ(before.path, after.path);
}

// The incremental clearance scan (father's ball / Lipschitz bound) returns exactly the full-scan
// distances, so the searches are identical, with far fewer map queries.
TEST(AstarClearance, IncrementalScanIsExact) {
    Planner &p = W->clr;
    const std::vector<std::pair<Vec3f, Vec3f>> queries = {
            {Vec3f(-5, -3, 1.5), Vec3f(5, -3, 1.5)},
            {Vec3f(-4, 4, 0.5),  Vec3f(6, -6, 2.5)},
            {Vec3f(-35, 0.9, 1.5), Vec3f(-25, 0.9, 1.5)},
            {Vec3f(-30, 3, 4.0), Vec3f(-26, 0.8, -1.0)},
            {Vec3f(25, -3, 1.5), Vec3f(35, 2, 1.5)},
    };
    for (const auto &wd: std::vector<std::pair<double, double>>{{1.0, 1.0}, {2.0, 1.4}, {0.5, 0.6}}) {
        p.astar->setClearanceCost(wd.first, wd.second);
        for (const auto &q: queries) {
            p.astar->setClearanceIncremental(false);
            Result full = search(p, q.first, q.second);
            p.astar->setClearanceIncremental(true);
            Result inc = search(p, q.first, q.second);
            ASSERT_EQ(full.ret, inc.ret);
            EXPECT_EQ(full.path, inc.path);
            EXPECT_EQ(full.stats.expanded, inc.stats.expanded);
            EXPECT_LT(inc.stats.clearance_probes, full.stats.clearance_probes);
        }
    }
    p.astar->setClearanceCost(0.0, 1.0);
}

// Flying along a surface: the stock path hugs it, with the cost the path moves out to clearance_dist
// from the inflated surface and still reaches the goal.
TEST(AstarClearance, SurfaceMovesAway) {
    Planner &p = W->clr;
    auto &sc = *W;
    const Vec3f s(-35, 0.9, 1.5), g(-25, 0.9, 1.5);

    p.astar->setClearanceCost(0.0, 0.0);
    Result stock = search(p, s, g);
    p.astar->setClearanceCost(1.0, 1.0);
    Result clr = search(p, s, g);

    ASSERT_EQ(stock.ret, path_search::REACH_GOAL);
    ASSERT_EQ(clr.ret, path_search::REACH_GOAL);
    EXPECT_LT((clr.path.back() - g).norm(), 0.2);
    const Clearance cs = pathClearance(stock.path, sc.kd);
    const Clearance cc = pathClearance(clr.path, sc.kd);
    std::cout << "[surface] stock: len " << pathLength(stock.path) << " clr min/med " << cs.min << "/" << cs.median
              << " exp " << stock.stats.expanded << " | w=1: len " << pathLength(clr.path) << " clr min/med "
              << cc.min << "/" << cc.median << " exp " << clr.stats.expanded << " probes "
              << clr.stats.clearance_probes << std::endl;
    EXPECT_LT(cs.median, 1.0);              // stock runs along y = 0.9
    EXPECT_GT(cc.median, cs.median + 0.4);  // the cost pushes it out (to ~1.5 m from the points)
    EXPECT_GE(cc.min, cs.min - 1e-6);       // never closer than stock
    EXPECT_LT(pathLength(clr.path), 1.3 * pathLength(stock.path));
}

// A wall with a 3 m slot: the stock path cuts the slot corner, the cost centres it.
TEST(AstarClearance, GapIsCentred) {
    Planner &p = W->clr;
    auto &sc = *W;
    const Vec3f s(-5, -3, 1.5), g(5, -3, 1.5);

    p.astar->setClearanceCost(0.0, 0.0);
    Result stock = search(p, s, g);
    p.astar->setClearanceCost(1.0, 1.0);
    Result clr = search(p, s, g);

    ASSERT_EQ(stock.ret, path_search::REACH_GOAL);
    ASSERT_EQ(clr.ret, path_search::REACH_GOAL);
    EXPECT_LT((clr.path.back() - g).norm(), 0.2);
    const Vec3f xs = crossing(stock.path), xc = crossing(clr.path);
    const double ds = clearanceAt(xs, sc.kd), dc = clearanceAt(xc, sc.kd);
    std::cout << "[gap] stock crosses at y " << xs.y() << " (clearance " << ds << "), w=1 at y " << xc.y()
              << " (clearance " << dc << ")" << std::endl;
    EXPECT_GT(dc, ds + 0.4);
    EXPECT_GT(dc, 1.2);  // slot half-width is 1.5 m
    EXPECT_GE(pathClearance(clr.path, sc.kd).min, pathClearance(stock.path, sc.kd).min - 1e-6);
}

// The cost never blocks: a slot whose inflated free width is a few cells is still taken, even with a
// heavy weight, because there is no other way through.
TEST(AstarClearance, NarrowGapStillTaken) {
    Planner &p = W->clr;
    const Vec3f s(25, -3, 1.5), g(35, -3, 1.5);
    for (double w: {0.0, 1.0, 5.0}) {
        p.astar->setClearanceCost(w, 1.0);
        Result r = search(p, s, g);
        ASSERT_EQ(r.ret, path_search::REACH_GOAL) << "w = " << w;
        EXPECT_LT((r.path.back() - g).norm(), 0.2);
        // the slot is at x = 30
        Vec3f x = r.path.front();
        for (const auto &q: r.path) {
            if (std::abs(q.x() - 30) < std::abs(x.x() - 30)) {
                x = q;
            }
        }
        EXPECT_LT(std::abs(x.y()), 0.35) << "w = " << w;
    }
}

// The planning horizon stays a path LENGTH with the cost on (distance_score is then a cost).
TEST(AstarClearance, HorizonIsPathLength) {
    Planner &p = W->clr;
    const Vec3f s(-38, 0.9, 1.5), g(-22, 0.9, 1.5);
    const double horizon = 4.0;
    for (double w: {0.0, 2.0}) {
        p.astar->setClearanceCost(w, 1.0);
        Result r = search(p, s, g, horizon);
        ASSERT_EQ(r.ret, path_search::REACH_HORIZON) << "w = " << w;
        const double len = pathLength(r.path);
        EXPECT_GT(len, horizon) << "w = " << w;
        EXPECT_LT(len, horizon + 0.5) << "w = " << w;
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    rclcpp::init(argc, argv);
    buildWorld();
    const int ret = RUN_ALL_TESTS();
    rclcpp::shutdown();
    return ret;
}
