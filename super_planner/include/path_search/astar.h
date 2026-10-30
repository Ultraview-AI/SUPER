/**
* This file is part of SUPER
*
* Copyright 2025 Yunfan REN, MaRS Lab, University of Hong Kong, <mars.hku.hk>
* Developed by Yunfan REN <renyf at connect dot hku dot hk>
* for more information see <https://github.com/hku-mars/SUPER>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* SUPER is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* SUPER is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with SUPER. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once

#include "Eigen/Dense"
#include "vector"
#include "rog_map_ros/rog_map_ros1.hpp"
#include "rog_map_ros/rog_map_ros2.hpp"
#include "queue"
#include "path_search/config.hpp"
#include "utils/header/type_utils.hpp"
#include <ros_interface/ros_interface.hpp>


namespace path_search {
    using namespace super_utils;

    constexpr double inf = 1 >> 20;
    struct GridNode;
    typedef GridNode *GridNodePtr;

    struct GridNode {
        enum enum_state : uint8_t {
            OPENSET = 1,
            CLOSEDSET = 2,
            UNDEFINED = 3
        } state{UNDEFINED};
        // Clearance of this node when it was expanded (clearance cost on only): squared distance
        // [cells^2] to the nearest inflated-occupied cell, or CLR_FREE when none is closer than
        // clearance_dist. Its children start their scan from it. Fits in former padding.
        static constexpr uint8_t CLR_FREE = 255;
        uint8_t clr_sq{CLR_FREE};

        int rounds{0};
        rog_map::Vec3i id_g;
        // Geometric path length from the start [cells]. Lives in what was alignment padding, so
        // GridNode stays 56 bytes (the A* buffer holds map_voxel_num nodes, 25 M on the Aegis).
        // Only maintained (and used, for the horizon) when the clearance cost is on; with it off
        // distance_score is the path length, as in stock SUPER.
        float path_len{0.f};
        double total_score{inf}, distance_score{inf};
        double distance_to_goal{inf};
        GridNodePtr father_ptr{nullptr};
    };
    static_assert(sizeof(void *) != 8 || sizeof(GridNode) == 56,
                  "GridNode grew: the A* node buffer is map_voxel_num nodes, every byte costs ~25 MB on the Aegis");

    class NodeComparator {
    public:
        bool operator()(GridNodePtr node1, GridNodePtr node2) {
            return node1->total_score > node2->total_score;
        }
    };

    class FrontierComparator {
    public:
        bool operator()(GridNodePtr node1, GridNodePtr node2) {
            return node1->distance_to_goal > node2->distance_to_goal;
        }
    };

    const int ON_INF_MAP = (1 << 0);
    const int ON_PROB_MAP = (1 << 1);
    const int UNKNOWN_AS_OCCUPIED = (1 << 3);
    const int UNKNOWN_AS_FREE = (1 << 4);
    const int USE_INF_NEIGHBOR = (1 << 5);
    const int DONT_USE_INF_NEIGHBOR = (1 << 6);

    class Astar {

        rog_map::ROGMapROS::Ptr map_ptr_;
        ros_interface::RosInterface::Ptr ros_ptr_;

        PathSearchConfig cfg_;
        const double tie_breaker_ = 1.0 + 1e-5;
        rog_map::vec_Vec3i sorted_pts;
        rog_map::vec_Vec3i neighbor_list;

        vector<GridNodePtr> grid_node_buffer_;

        int rounds_{0};

        // Clearance cost: cell offsets closer than clearance_dist, sorted by distance, built for
        // clearance_offsets_res_ (the search resolution).
        struct ClearanceOffsets {
            rog_map::vec_Vec3i off;
            std::vector<int> sq;  // squared length [cells^2], ascending
        };
        // The ball (|o| < clearance_dist), and per step direction the "crescent": the part of a
        // child's ball that is outside its father's ball, so a child of a father with a free ball
        // only scans the crescent.
        ClearanceOffsets clearance_ball_;
        ClearanceOffsets clearance_crescent_[27];
        double clearance_offsets_res_{-1.0}, clearance_offsets_dist_cfg_{-1.0};
        long clearance_probe_cnt_{0};
        bool clearance_incremental_{true};

        bool clearanceCostEnabled() const;

        void buildClearanceOffsets();

        /// Exact distance [m] from `node` to the nearest inflated-occupied cell, capped at
        /// clearance_dist; also stores it in node->clr_sq. Uses the father's stored clearance
        /// (1-Lipschitz) to skip cells that cannot be occupied.
        double getClearance(GridNodePtr node);

        static constexpr int DIAG = 0;
        static constexpr int MANH = 1;
        static constexpr int EUCL = 2;

        struct MissionData {
            rog_map::Vec3f start_pt;
            rog_map::Vec3f goal_pt;
            double searching_horizon;
            bool use_inf_map{false};
            bool use_prob_map{false};
            bool unknown_as_occ{false};
            bool unknown_as_free{false};
            bool use_inf_neighbor{false};
            double resolution;
            rog_map::Vec3i local_map_center_id_g;
            rog_map::Vec3f local_map_center_d;
            double mission_rcv_WT{0};
            rog_map::Vec3f local_map_max_d, local_map_min_d;
            std::mutex mission_mtx;
        } md_;



        double getHeu(GridNodePtr node1, GridNodePtr node2, int type = DIAG) const;

         int getLocalIndexHash(const rog_map::Vec3i &id_in) const;

        void posToGlobalIndex(const rog_map::Vec3f &pos, rog_map::Vec3i &id_g) const ;

        void globalIndexToPos(const rog_map::Vec3i &id_g, rog_map::Vec3f &pos) const;

        bool insideLocalMap(const rog_map::Vec3f &pos) const;

        bool insideLocalMap(const rog_map::Vec3i &id_g) const;

        bool neighborHaveOne(const rog_map::GridType &type, const rog_map::Vec3i &src_id);

        RET_CODE setup(const rog_map::Vec3f &start_pt, const rog_map::Vec3f &goal_pt, const int &flag,
                       const double &searching_horizon = 9999);

        void retrievePath(GridNodePtr current, vector<GridNodePtr> &path);

        void ConvertNodePathToPointPath(const vector<GridNodePtr> &node_path, rog_map::vec_Vec3f &point_path);

    public:

        Astar(const std::string & cfg_path,
              const ros_interface::RosInterface::Ptr &ros_ptr,
              rog_map::ROGMapROS::Ptr rm);

        ~Astar() {};

        typedef std::shared_ptr<Astar> Ptr;

        void setVisualProcessEn(const bool &en);

        void setFineInfNeighbors(const int & neighbor_step);

        /// Override astar/clearance_weight [1/m] and astar/clearance_dist [m]; weight 0 = stock search.
        void setClearanceCost(const double &weight, const double &dist);

        /// Tests only: false makes every clearance query a full ball scan (the reference).
        void setClearanceIncremental(const bool &en) { clearance_incremental_ = en; }

        /// Statistics of the last pointToPointPathSearch call (for tests / benchmarks).
        struct SearchStats {
            int expanded{0};
            long clearance_probes{0};
        };

        SearchStats getLastSearchStats() const { return last_stats_; }

    private:
        SearchStats last_stats_;

    public:

        RET_CODE pointToPointPathSearch(const rog_map::Vec3f &start_pt, const rog_map::Vec3f &end_pt,
                                        const int &flag,
                                        const double &searching_horizon,
                                        rog_map::vec_Vec3f &out_path,
                                        const double &time_out = 0.1);

        /// @ brief: The escape path only for path search from prob map to inf map. from non-occupied point to
        ///          inf map free (or known freee) point . Aim to find a path from current point to (known) free point
        /// @ param:
        RET_CODE escapePathSearch(const rog_map::Vec3f &start_pt, const int flag, rog_map::vec_Vec3f &out_path);


    };
}