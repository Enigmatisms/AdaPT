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
    def aabb_test(self, inv_ray: vec3, ray_o: vec3):
        """ AABB used to skip some of the objects """
        t_min = (self.mini - ray_o) * inv_ray
        t_max = (self.maxi - ray_o) * inv_ray
        t_near = ti.min(t_min, t_max).max()
        t_far = ti.max(t_min, t_max).min()
        return (t_near < t_far) and t_far > 0, t_near
    
    @ti.func
    def get_info(self):
        return self.obj_idx, self.prim_idx
    
@ti.dataclass
class LinearNode:
    mini:       vec3
    maxi:       vec3
    base:       int
    prim_cnt:   int
    all_offset: int

    @ti.func
    def aabb_test(self, inv_ray: vec3, ray_o: vec3):
        """ AABB used to skip some of the objects """
        t_min = (self.mini - ray_o) * inv_ray
        t_max = (self.maxi - ray_o) * inv_ray
        t_near = ti.min(t_min, t_max).max()
        t_far = ti.max(t_min, t_max).min()
        return (t_near < t_far) and t_far > 0, t_near
    
    @ti.func
    def is_leaf(self):
        return self.all_offset == 1
    
    @ti.func
    def get_range(self):
        return self.base, self.base + self.prim_cnt
