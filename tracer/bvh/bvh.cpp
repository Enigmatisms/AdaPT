#include <algorithm>
#include "bvh.h"

namespace py = pybind11;
using IntPair = std::pair<int, int>;

constexpr size_t num_bins = 12;            // the same as PBR-book 
constexpr float traverse_cost = 0.1;
constexpr float max_node_prim = 1;

enum SplitAxis: int {AXIS_X, AXIS_Y, AXIS_Z, AXIS_NONE};

struct AABB {
    Eigen::Vector3f mini;
    Eigen::Vector3f maxi;

    AABB(): mini(Eigen::Vector3f::Zero()), maxi(Eigen::Vector3f::Zero()) {}
    /** Review needed: is this good to implement two overload? */
    AABB(const Eigen::Vector3f& mini, const Eigen::Vector3f& maxi): mini(mini), maxi(maxi) {}
    AABB(Eigen::Vector3f&& mini, Eigen::Vector3f&& maxi): mini(mini), maxi(maxi) {}
    AABB(const Eigen::Matrix3f& primitive, bool is_sphere = false) {
        if (is_sphere) {
            mini = primitive.col(0) - primitive.col(1);
            maxi = primitive.col(0) + primitive.col(1);
        } else {
            mini = primitive.rowwise().minCoeff();      // rowwise --- get coeff in each row
            maxi = primitive.rowwise().maxCoeff();
        }
    }

    AABB operator+(const AABB& aabb) const {
        return AABB(aabb.mini.cwiseMax(mini), aabb.maxi.cwiseMax(maxi));
    }

    AABB& operator+=(const AABB& aabb) {
        this->mini = aabb.mini.cwiseMin(this->mini);
        this->maxi = aabb.mini.cwiseMax(this->mini);
    }

    void clear() {
        mini.setZero();
        maxi.setZero();
    }

    float area() const {
        Eigen::Vector3f diff = maxi - mini;
        return 2. * (diff(0) * diff(1) + diff(1) * diff(2) + diff(0) * diff(2));
    }
};

struct BVHInfo {
    // BVH is for both triangle meshes and spheres
    AABB bound;
    Eigen::Vector3f centroid;
    int prim_idx;
    int obj_idx;

    BVHInfo(): centroid(Eigen::Vector3f::Zero()), prim_idx(-1), obj_idx(-1) {}
    BVHInfo(const Eigen::Matrix3f& primitive, int prim_idx, int obj_idx, bool is_sphere = false): 
        bound(primitive), prim_idx(prim_idx), obj_idx(obj_idx) 
    {
        // Extract two vertices for the primitive, once converted to AABB
        // We don't need to distinguish between mesh or sphere
        // Note that vertex vectors in the primitive matrix are col-major order
        if (is_sphere)
            centroid = primitive.col(0);
        else
            centroid = primitive.rowwise().mean();      // barycenter
    }
};

struct AxisBins {
    AABB bound;
    int prim_cnt;

    AxisBins(): prim_cnt(0.f) {}

    void push(const BVHInfo& bvh) {
        bound += bvh.bound;
        prim_cnt ++;
    }
};

class BVHNode {
public:
    BVHNode(): base(0), prim_num(0), axis(AXIS_NONE), lchild(nullptr), rchild(nullptr) {}
    ~BVHNode() {
        if (lchild != nullptr) delete lchild;
        if (rchild != nullptr) delete rchild;
    }
public:
    // The axis start and end are scaled up a little bit
    SplitAxis max_extent_axis(const std::vector<BVHInfo>& bvhs, std::vector<float>& bins) const;
public:
    int base;
    int prim_num;

    AABB bound;
    SplitAxis axis;
    BVHNode *lchild, *rchild;
};

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

// FIXME: there must be some other parameters to fill in, like the range of primitives 
// FIXME: root node creation
void recursive_bvh_SAH(BVHNode* const cur_node, std::vector<BVHInfo>& bvh_infos) {
    if (cur_node->prim_num > 4) {   // SAH
        // Step 1: decide the axis that expands the maximum extent of space
        std::vector<float> bins;        // bins: from (start_pos + interval) to end_pos
        SplitAxis max_axis = cur_node->max_extent_axis(bvh_infos, bins);

        // Step 2: binning the space
        const int prim_num = cur_node->prim_num, base = cur_node->base, max_pos = base + prim_num - 1;
        std::array<AxisBins, num_bins> idx_bins;
        for (int i = cur_node->base; i <= max_pos; i++) {
            size_t index = std::lower_bound(bins.begin(), bins.end(), bvh_infos[i].centroid[max_axis]) - bins.begin();
            idx_bins[index].push(bvh_infos[i]);
        }

        // Step 3: forward-backward linear sweep for heuristic calculation
        AABB fwd_bound, bwd_bound;
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
        int seg_idx = 0;                // this index is used for indexing variable `bins`
        float node_inv_area = 1. / cur_node->bound.area(), min_cost = 5e9, node_prim_cnt = float(cur_node->prim_num);
        // TODO: there is more to consider: whether to segement, and where to segment
        for (int i = 0; i < num_bins - 1; i++) {
            float cost = traverse_cost + node_inv_area * 
                (float(prim_cnts[i]) * fwd_areas[i] + (node_prim_cnt - (prim_cnts[i])) * bwd_areas[i]);
            if (cost < min_cost) {
                min_cost = cost;
                seg_idx = i;
            }
        }
        if (min_cost < node_prim_cnt) {         // cost of splitting is less than making this node a leaf node
            // Step 5: split the node and initialize the children
            cur_node->lchild = new BVHNode();
            cur_node->rchild = new BVHNode();
            int child_prim_cnt = prim_cnts[seg_idx];
            cur_node->lchild->base = base;
            cur_node->lchild->prim_num = child_prim_cnt;
            cur_node->rchild->base = base + child_prim_cnt;
            cur_node->rchild->prim_num = prim_num - child_prim_cnt;
            fwd_bound.clear();
            bwd_bound.clear();
            for (int i = 0; i < seg_idx; i++)
                fwd_bound += idx_bins[i].bound;
            for (int i = num_bins - 1; i > seg_idx; i--)
                bwd_bound += idx_bins[i].bound;
            cur_node->lchild->bound = fwd_bound;
            cur_node->rchild->bound = bwd_bound;
            cur_node->axis = max_axis;

            // Step 6: reordering the BVH info in the vector to make the segment contiguous 
            std::partition(bvh_infos.begin() + base, bvh_infos.begin() + base + prim_num,
                [pivot = bins[seg_idx], dim = max_axis](const BVHInfo& bvh) {
                    return bvh.centroid[dim] < pivot;
            });

            // Step 7: start recursive splitting for the children
            if (cur_node->lchild->prim_num > max_node_prim)
                recursive_bvh_SAH(cur_node->lchild, bvh_infos);
            if (cur_node->rchild->prim_num > max_node_prim)
                recursive_bvh_SAH(cur_node->rchild, bvh_infos);
        } else {
            // This is a leaf node, yet this is the only way that a leaf node contains more than one primitive
            cur_node->axis = AXIS_NONE;
        }
    } else {                        // equal primitive number

    }
}