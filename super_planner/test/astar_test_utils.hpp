// Shared helpers for the A* clearance-cost unit tests and benchmark: build a ROG-Map from a point
// cloud (load_pcd_en, no ROS callbacks, no raycasting) and an Astar on top of it.
#pragma once

#include <path_search/astar.h>
#include <ros_interface/ros2/ros2_interface.hpp>
#include <rog_map_ros/rog_map_ros2.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>

namespace astar_test {

    struct MapSpec {
        double resolution{0.2};
        double inflation_resolution{0.2};
        int inflation_step{2};
        Eigen::Vector3d map_size{30, 30, 8};
        Eigen::Vector3d origin{0, 0, 1.5};
        double virtual_ground{-5.0};
        double virtual_ceil{10.0};
        Eigen::Vector3i astar_vox{150, 150, 40};
        double clearance_weight{0.0};
        double clearance_dist{0.0};
        bool write_clearance_keys{true};
    };

    inline std::string tmpPath(const std::string &stem) {
        return "/tmp/astar_clr_" + std::to_string(getpid()) + "_" + stem;
    }

    // Writes the yaml read by both ROG-Map (rog_map/...) and the A* (astar/...).
    inline std::string writeConfig(const MapSpec &s, const std::string &pcd_path, const std::string &stem) {
        std::ostringstream y;
        y << "rog_map:\n"
          << "  resolution: " << s.resolution << "\n"
          << "  inflation_resolution: " << s.inflation_resolution << "\n"
          << "  inflation_step: " << s.inflation_step << "\n"
          << "  unk_inflation_en: false\n"
          << "  map_size: [" << s.map_size.x() << ", " << s.map_size.y() << ", " << s.map_size.z() << "]\n"
          << "  fix_map_origin: [" << s.origin.x() << ", " << s.origin.y() << ", " << s.origin.z() << "]\n"
          << "  frontier_extraction_en: false\n"
          << "  virtual_ceil_height: " << s.virtual_ceil << "\n"
          << "  virtual_ground_height: " << s.virtual_ground << "\n"
          << "  load_pcd_en: true\n"
          << "  pcd_name: \"" << pcd_path << "\"\n"
          << "  map_sliding:\n    enable: false\n    threshold: 0.3\n"
          << "  esdf:\n    enable: false\n    resolution: 0.2\n    local_update_box: [5, 5, 3]\n"
          << "  ros_callback:\n    enable: false\n    cloud_topic: \"/x\"\n    odom_topic: \"/y\"\n    odom_timeout: 2.0\n"
          << "  visualization:\n    enable: false\n    time_rate: 0\n    frame_rate: 0\n    range: [1, 1, 1]\n"
          << "    frame_id: \"world\"\n    pub_unknown_map_en: false\n"
          << "  intensity_thresh: -10\n"
          << "  point_filt_num: 1\n"
          << "  raycasting:\n    enable: false\n    batch_update_size: 1\n    local_update_box: [100, 100, 20]\n"
          << "    ray_range: [0.5, 100]\n    p_min: 0.12\n    p_miss: 0.49\n    p_free: 0.499\n"
          << "    p_occ: 0.85\n    p_hit: 0.9\n    p_max: 0.98\n    unk_thresh: 1.0\n"
          << "astar:\n"
          << "  map_voxel_num: [" << s.astar_vox.x() << ", " << s.astar_vox.y() << ", " << s.astar_vox.z() << "]\n"
          << "  visual_process: false\n"
          << "  allow_diag: true\n"
          << "  heu_type: 2\n"
          << "  debug_visualization_en: false\n";
        if (s.write_clearance_keys) {
            y << "  clearance_weight: " << s.clearance_weight << "\n"
              << "  clearance_dist: " << s.clearance_dist << "\n";
        }
        const std::string path = tmpPath(stem + ".yaml");
        std::ofstream(path) << y.str();
        return path;
    }

    struct Planner {
        rog_map::ROGMapROS::Ptr map;
        ros_interface::RosInterface::Ptr ros;
        path_search::Astar::Ptr astar;
    };

    inline Planner makePlanner(const rclcpp::Node::SharedPtr &node, const std::string &cfg) {
        Planner p;
        p.map = std::make_shared<rog_map::ROGMapROS>(node, cfg);
        p.ros = std::make_shared<ros_interface::Ros2Interface>(node);
        p.astar = std::make_shared<path_search::Astar>(cfg, p.ros, p.map);
        return p;
    }

    // Flags SuperPlanner uses for its primary (inflated-map) front-end search.
    inline int frontendFlags() {
        return path_search::ON_INF_MAP | path_search::UNKNOWN_AS_FREE | path_search::DONT_USE_INF_NEIGHBOR;
    }

    // Distance from each path point to the nearest raw cloud point.
    struct Clearance {
        double min{0}, median{0}, mean{0};
    };

    inline Clearance pathClearance(const rog_map::vec_Vec3f &path, pcl::KdTreeFLANN<pcl::PointXYZI> &kd) {
        Clearance c;
        if (path.empty()) {
            return c;
        }
        std::vector<double> d;
        std::vector<int> idx(1);
        std::vector<float> sq(1);
        for (const auto &p: path) {
            pcl::PointXYZI q;
            q.x = p.x();
            q.y = p.y();
            q.z = p.z();
            kd.nearestKSearch(q, 1, idx, sq);
            d.push_back(std::sqrt(sq[0]));
        }
        double sum = 0;
        for (double v: d) {
            sum += v;
        }
        c.mean = sum / d.size();
        std::sort(d.begin(), d.end());
        c.min = d.front();
        c.median = d[d.size() / 2];
        return c;
    }

    inline double pathLength(const rog_map::vec_Vec3f &path) {
        double l = 0;
        for (size_t i = 1; i < path.size(); i++) {
            l += (path[i] - path[i - 1]).norm();
        }
        return l;
    }
}
