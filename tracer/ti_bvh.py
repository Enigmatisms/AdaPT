"""
    Taichi BVH part: exporing python BVH to taichi
    @author: Qianyue He
    @date: 2023.4.10
"""

import taichi as ti
from taichi.math import vec3

@ti.dataclass
class LinearBVH:
    mini:       vec3
    maxi:       vec3
    obj_idx:    int
    prim_idx:   int

    @ti.func
    def aabb_test(self, ray: vec3, ray_o: vec3):
        """ AABB used to skip some of the objects """
        t_min = (self.mini - ray_o) / ray
        t_max = (self.maxi - ray_o) / ray
        t_near = ti.min(t_min, t_max).max()
        t_far = ti.max(t_min, t_max).min()
        return (t_near < t_far) and t_far > 0, t_near, t_far
    
@ti.dataclass
class LinearNode:
    mini:       vec3
    maxi:       vec3
    base:       int
    prim_cnt:   int
    all_offset: int

    @ti.func
    def aabb_test(self, ray: vec3, ray_o: vec3):
        """ AABB used to skip some of the objects """
        t_min = (self.mini - ray_o) / ray
        t_max = (self.maxi - ray_o) / ray
        t_near = ti.min(t_min, t_max).max()
        t_far = ti.max(t_min, t_max).min()
        return (t_near < t_far) and t_far > 0, t_near, t_far
    
    @ti.func
    def is_leaf(self):
        return self.all_offset == 1
    
def export_python_bvh(ti_nodes: ti.template(), ti_bvhs: ti.template(), lin_nodes: list, lin_bvhs: list):
    for i, node in enumerate(lin_nodes):
        ti_nodes[i] = LinearNode(mini = vec3(node.mini), maxi = vec3(node.maxi),
            base = node.base, prim_cnt = node.prim_cnt, all_offset = node.all_offset
        )
    for i, bvh in enumerate(lin_bvhs):
        ti_bvhs[i] = LinearBVH(mini = vec3(bvh.mini), maxi = vec3(bvh.maxi), obj_idx = bvh.obj_idx, prim_idx = bvh.prim_idx)
