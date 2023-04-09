#include <stack>
#include <algorithm>
#include "bvh.h"
#include "bvh_helper.h"

using IntPair = std::pair<int, int>;

constexpr size_t num_bins = 12;            // the same as PBR-book 
constexpr float traverse_cost = 0.1;
constexpr float max_node_prim = 1;

SplitAxis BVHNode::max_extent_axis(const std::vector<BVHInfo>& bvhs, std::vector<float>& bins) const {
    Eigen::Vector3f min_ctr = bvhs[base].centroid, max_ctr = bvhs[base].centroid;
    for (int i = 1; i < prim_num; i++) {
        const Eigen::Vector3f& ctr = bvhs[base + i].centroid;
        min_ctr = min_ctr.cwiseMin(ctr);
        max_ctr = min_ctr.cwiseMax(ctr);
    }
    Eigen::Vector3f diff = max_ctr - min_ctr;
    float max_diff = diff(0);
    int split_axis = 0;
    for (int i = 1; i < 3; i++) {
        if (diff(i) > max_diff) {
            max_diff = diff(i);
            split_axis = i;
        }
    }
    bins.resize(num_bins);
    float min_r = min_ctr(split_axis) - 0.001f, interval = (max_diff + 0.002f) / float(num_bins);
    std::transform(bins.begin(), bins.end(), bins.begin(), [min_r, interval, i = 0] (const float&) mutable {
        i++; return min_r + interval * float(i);
    });
    return SplitAxis(split_axis);
}

// Convert numpy wavefront obj to eigen vector
void wavefront_input(const py::array_t<float>& obj_array, std::vector<Eigen::Matrix3f>& meshes) {
    auto buf = obj_array.request();
    auto result_shape = obj_array.shape();
    const float* const ptr = (float*)buf.ptr;
    size_t num_triangles = result_shape[0];
    std::vector<Eigen::Matrix3f> results(num_triangles, Eigen::Matrix3f::Zero());

    for (size_t i = 0; i < num_triangles; i++) {
        Eigen::Matrix3f mat = Eigen::Matrix3f::Zero();
        float* const mat_data = mat.data();
        size_t base_index = i * 9;
        for (size_t j = 0; j < 9; j++) {
            mat_data[j] = ptr[base_index + j];
        }
        
    }
}

void index_input(const py::array_t<int>& obj_idxs, const py::array_t<int>& sphere_flags, std::vector<IntPair>& idxs) {
    size_t result_shape = obj_idxs.shape()[0];
    auto idx_buf  = obj_idxs.request();
    auto flag_buf = sphere_flags.request();
    const int* const idx_ptr  = (int*)idx_buf.ptr;
    const int* const flag_ptr = (int*)flag_buf.ptr;

    idxs.resize(result_shape);
    std::transform(idx_ptr, idx_ptr + result_shape, flag_ptr, idxs.begin(), 
        [] (int obj_idx, int sphere_flag) {return std::make_pair(obj_idx, sphere_flag);}
    );
}

void create_bvh_info(const std::vector<Eigen::Matrix3f>& meshes, const std::vector<IntPair>& idxs, std::vector<BVHInfo>& bvh_infos) {
    bvh_infos.resize(meshes.size());
    std::transform(meshes.begin(), meshes.end(), idxs.begin(), bvh_infos.begin(),
        [i = 0] (const Eigen::Matrix3f& primitive, const IntPair& idx_info) mutable {
            return BVHInfo(primitive, i++, idx_info.first, idx_info.second > 0);
        }
    ); 
}

BVHNode* bvh_root_start(const py::array_t<float>& world_min, const py::array_t<float>& world_max, int& node_num, std::vector<BVHInfo>& bvh_infos) {
    // Build BVH tree root node and start recursive tree construction
    BVHNode* root_node = new BVHNode(0, bvh_infos.size());
    Eigen::Vector3f &bound_min = root_node->bound.mini, &bound_max = root_node->bound.maxi;
    auto min_buf = world_min.request(), max_buf = world_max.request();
    const float* const min_ptr = (float*)min_buf.ptr, * const max_ptr = (float*)max_buf.ptr;
    for (int i = 0; i < 3; i++) {
        bound_min(i) = min_ptr[i];
        bound_max(i) = max_ptr[i];
    }
    node_num = recursive_bvh_SAH(root_node, bvh_infos);
    return root_node;
}

int recursive_bvh_SAH(BVHNode* const cur_node, std::vector<BVHInfo>& bvh_infos) {
    AABB fwd_bound, bwd_bound;
    int seg_idx = 0, child_prim_cnt = 0;                // this index is used for indexing variable `bins`
    const int prim_num = cur_node->prim_num, base = cur_node->base, max_pos = base + prim_num;
    float min_cost = 5e9, node_prim_cnt = float(cur_node->prim_num), node_inv_area = 1. / cur_node->bound.area();

    // Step 1: decide the axis that expands the maximum extent of space
    std::vector<float> bins;        // bins: from (start_pos + interval) to end_pos
    SplitAxis max_axis = cur_node->max_extent_axis(bvh_infos, bins);
    if (cur_node->prim_num > 4) {   // SAH

        // Step 2: binning the space
        std::array<AxisBins, num_bins> idx_bins;
        for (int i = cur_node->base; i < max_pos; i++) {
            size_t index = std::lower_bound(bins.begin(), bins.end(), bvh_infos[i].centroid[max_axis]) - bins.begin();
            idx_bins[index].push(bvh_infos[i]);
        }

        // Step 3: forward-backward linear sweep for heuristic calculation
        std::array<int, num_bins> prim_cnts;
        std::array<float, num_bins> fwd_areas, bwd_areas;
        for (int i = 0; i < num_bins; i++) {
            fwd_bound   += idx_bins[i].bound;
            prim_cnts[i] = idx_bins[i].prim_cnt;
            fwd_areas[i] = fwd_bound.area();
            if (i > 0) {
                bwd_bound += idx_bins[num_bins - 1].bound;
                bwd_areas[num_bins - 1 - i] = bwd_bound.area();
            }
        }
        std::partial_sum(prim_cnts.begin(), prim_cnts.end(), prim_cnts.begin());

        // Step 4: use the calculated area to computed the segment boundary
        float ;
        int seg_bin_idx = 0;
        for (int i = 0; i < num_bins - 1; i++) {
            float cost = traverse_cost + node_inv_area * 
                (float(prim_cnts[i]) * fwd_areas[i] + (node_prim_cnt - (prim_cnts[i])) * bwd_areas[i]);
            if (cost < min_cost) {
                min_cost = cost;
                seg_bin_idx = i;
            }
        }
        // Step 5: reordering the BVH info in the vector to make the segment contiguous (partition around pivot)
        if (min_cost < node_prim_cnt) {
            std::partition(bvh_infos.begin() + base, bvh_infos.begin() + max_pos,
                [pivot = bins[seg_bin_idx], dim = max_axis](const BVHInfo& bvh) {
                    return bvh.centroid[dim] < pivot;
            });
            child_prim_cnt = prim_cnts[seg_bin_idx];
            seg_idx = base + child_prim_cnt;        // bvh[seg_idx] will be in rchild
        }
        
        fwd_bound.clear();
        bwd_bound.clear();
        for (int i = 0; i < seg_idx; i++)       // calculate child node bound
            fwd_bound += idx_bins[i].bound;
        for (int i = num_bins - 1; i > seg_idx; i--)
            bwd_bound += idx_bins[i].bound;
    } else {                                    // equal primitive number
        seg_idx = (base + max_pos) >> 1;
        // Step 5: reordering the BVH info in the vector to make the segment contiguous (keep around half of the bvh in lchild)
        std::nth_element(bvh_infos.begin() + base, bvh_infos.begin() + seg_idx, bvh_infos.begin() + max_pos,
            [dim = max_axis] (const BVHInfo& bvh1, const BVHInfo& bvh2) {
                return bvh1.centroid[dim] < bvh2.centroid[dim];
            }
        );
        for (int i = base; i < seg_idx; i++)    // calculate child node bound
            fwd_bound += bvh_infos[i].bound;
        for (int i = seg_idx; i < max_pos; i--)
            bwd_bound += bvh_infos[i].bound;
        child_prim_cnt = seg_idx - base;        // bvh[seg_idx] will be in rchild
        float split_cost = traverse_cost + node_inv_area * 
                (fwd_bound.area() * child_prim_cnt + bwd_bound.area() * (node_prim_cnt - child_prim_cnt));
        if (split_cost >= node_prim_cnt)
            child_prim_cnt = 0;
    }
    if (child_prim_cnt > 0) {             // cost of splitting is less than making this node a leaf node
        // Step 5: split the node and initialize the children
        cur_node->lchild = new BVHNode(base, child_prim_cnt);
        cur_node->rchild = new BVHNode(base + child_prim_cnt, prim_num - child_prim_cnt);

        cur_node->lchild->bound = fwd_bound;
        cur_node->rchild->bound = bwd_bound;
        cur_node->axis = max_axis;
        // Step 7: start recursive splitting for the children
        int node_num = 1;
        if (cur_node->lchild->prim_num > max_node_prim)
            node_num += recursive_bvh_SAH(cur_node->lchild, bvh_infos);
        else node_num ++;
        if (cur_node->rchild->prim_num > max_node_prim)
            node_num += recursive_bvh_SAH(cur_node->rchild, bvh_infos);
        else node_num ++;
        return node_num;
    } else {
        // This is a leaf node, yet this is the only way that a leaf node contains more than one primitive
        cur_node->axis = AXIS_NONE;
        return 1;
    }
}

// This is the final function call for `bvh_build`
int recursive_linearize(BVHNode* cur_node, std::vector<LinearNode>& lin_nodes) {
    // BVH tree should be linearized to better traverse and fit in the system memory
    // The linearized BVH tree should contain: bound, base, prim_cnt, rchild_offset, total_offset (to skip the entire node)
    // Note that if rchild_offset is -1, then the node is leaf. Leaf node points to primitive array
    // which is already sorted during BVH construction, containing primitive_id and obj_id for true intersection
    // Note that lin_nodes has been reserved
    size_t current_size = lin_nodes.size();
    lin_nodes.emplace_back(cur_node);
    if (cur_node->lchild != nullptr) {
        int lnodes = recursive_linearize(cur_node->lchild, lin_nodes);
        lin_nodes[current_size].rc_offset = lnodes + 1;
        lnodes += recursive_linearize(cur_node->rchild, lin_nodes);
        lin_nodes[current_size].all_offset = lnodes + 1;
        return lnodes + 1;                      // include the cur_node                       
    } else {
        lin_nodes.back().rc_offset = 0;         // no rchild, therefore 0
        lin_nodes.back().all_offset = 1;        // to skip the current sub-tree, index should just add 1
        return 1;
    }
}

void bvh_build(const py::array_t<float>& obj_array, const py::array_t<int>& id_array) {
    // TODO: input output should be considered, about object id, sphere_flag and primitive for bounds calculation
    // The output should be reordered BVHInfo (this should be a pybind class, too?) and linearized BVH tree nodes
}
