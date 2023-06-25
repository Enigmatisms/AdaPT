"""
    Path tracer for indirect / global illumination
    This module will be progressively built. Currently, participating media is not supported
    @author: Qianyue He
    @date: 2023.1.26
"""

import sys
sys.path.append("..")

import numpy as np
import taichi as ti
import taichi.math as tm
from taichi.math import vec2, vec3, mat3

from typing import List
from la.cam_transform import *

from parsers.obj_desc import ObjDescriptor
from tracer.interaction import Interaction
from parsers.xml_parser import scene_parsing
from renderer.constants import INV_PI, INV_2PI

from rich.console import Console
CONSOLE = Console(width = 128)

__eps__ = 1e-4
__inv_eps__ = 1 - __eps__ * 2.

@ti.data_oriented
class TracerBase:
    """
        Simple Ray tracing using Bary-centric coordinates
        This tracer can yield result with global illumination effect
    """
    def __init__(self, objects: List[ObjDescriptor], prop: dict):
        self.w          = prop['film']['width']                              # image is a standard square
        self.h          = prop['film']['height']
        # center crop (for a small patch of )
        self.crop_x     = prop['film'].get('crop_x', 0)
        self.crop_y     = prop['film'].get('crop_y', 0)
        self.crop_rx    = prop['film'].get('crop_rx', 0)
        self.crop_ry    = prop['film'].get('crop_ry', 0)
        self.do_crop    = (self.crop_rx > 0) and (self.crop_ry > 0)

        if self.do_crop:
            self.start_x = self.crop_x - self.crop_rx
            self.end_x   = self.crop_x + self.crop_rx
            self.start_y = self.crop_y - self.crop_ry
            self.end_y   = self.crop_y + self.crop_ry
        else:
            self.start_x = 0
            self.start_y = 0
            self.end_x   = self.w
            self.end_y   = self.h

        self.max_bounce = prop['max_bounce']
        self.use_rr     = prop['use_rr']

        self.anti_alias         = False
        self.stratified_sample  = False

        self.focal      = fov2focal(prop['fov'], min(self.w, self.h))
        self.inv_focal  = 1. / self.focal
        self.half_w     = self.w / 2
        self.half_h     = self.h / 2

        self.num_objects = len(objects)
        self.num_prims  = sum([obj.tri_num for obj in objects])

        self.cam_orient = prop['transform'][0]                          # first field is camera orientation
        self.cam_orient /= np.linalg.norm(self.cam_orient)
        self.cam_t      = vec3(prop['transform'][1])
        self.cam_r      = mat3(np_rotation_between(np.float32([0, 0, 1]), self.cam_orient))
        
        # A more compact way to store the primitives
        self.aabbs      = ti.Vector.field(3, float, (self.num_objects, 2))

        # geometric normal for surfaces
        self.normals    = ti.Vector.field(3, float)
        self.prims      = ti.Vector.field(3, float)                             # leveraging SSDS, shape (N, mesh_num, 3) - vector3d
        self.uv_coords  = ti.Vector.field(2, float)                             # uv coordinates
        self.precom_vec = ti.Vector.field(3, float)
        self.v_normals  = ti.Vector.field(3, float)
        self.pixels     = ti.Vector.field(3, float, (self.w, self.h))           # output: color

        self.dense_nodes = ti.root.dense(ti.i, self.num_prims)
        self.dense_nodes.place(self.normals)

        self.prim_handle = self.dense_nodes.dense(ti.j, 3)
        self.prim_handle.place(self.prims, self.precom_vec)      # for simple shapes, this would be efficient
        self.prim_handle.place(self.uv_coords)
        self.has_v_normal = prop["has_vertex_normal"]
        if prop["has_vertex_normal"]:
            CONSOLE.log(f"Vertex normals found. Allocating storage for v-normals.")
            self.prim_handle.place(self.v_normals)                  # actual vertex normal storage
        else:
            ti.root.bitmasked(ti.i, 1).place(self.v_normals)        # A dummy place

        # pos0: start_idx, pos1: number of primitives, pos2: obj_id (being triangle / sphere? Others to be added, like cylinder, etc.)
        self.obj_info  = ti.field(int, (self.num_objects, 3))
        self.cnt       = ti.field(int, ())                          # useful in path tracer (sample counter)

    def get_check_point(self):
        """Only offered in PathTracer"""
        raise NotImplementedError("Checkpoint saver is not implemented in TracerBase.")

    def __repr__(self):
        """
            For debug purpose
        """
        return f"tracer_base: number of object {self.num_objects}, w, h: ({self.w}, {self.h}). Focal: {self.focal}"

    def initialze(self, _objects: List[ObjDescriptor]):
        pass

    @ti.func
    def pix2ray(self, i, j):
        """
            Convert pixel coordinate to ray direction
            For pinhole camera model, rays can be precomputed, therefore not useful
            - anti_alias: whether to use pixel sample jittering for anti-aliasing
            - str_sample: whether to use stratified sampling 
        """
        pi = float(i)
        pj = float(j)
        vx = 0.5
        vy = 0.5
        if ti.static(self.anti_alias):
            if ti.static(self.stratified_sample): # sequential stratified sampling
                mod_val = self.cnt[None] % 16
                vx = float(mod_val % 4)   * 0.25 + ti.random(float) * 0.25
                vy = float(mod_val // 4) * 0.25 + ti.random(float) * 0.25
            else:    # uniform sampling
                vx = ti.random(float) * __inv_eps__ + __eps__
                vy = ti.random(float) * __inv_eps__ + __eps__
        cam_dir = vec3([(self.half_w + vx - pi) * self.inv_focal, (pj - self.half_h - vy) * self.inv_focal, 1.])
        return (self.cam_r @ cam_dir).normalized()

    @ti.func
    def aabb_test(self, aabb_idx, ray: vec3, ray_o: vec3):
        """ AABB used to skip some of the objects """
        t_min = (self.aabbs[aabb_idx, 0] - ray_o) / ray
        t_max = (self.aabbs[aabb_idx, 1] - ray_o) / ray
        t_near = ti.min(t_min, t_max).max()
        t_far = ti.max(t_min, t_max).min()
        return (t_near < t_far) and t_far > 0, t_near, t_far

    @ti.func
    def ray_intersect(self, ray, start_p, min_depth = -1.0):
        """ Villina intersection logic without acceleration structure """
        obj_id = -1
        prm_id = -1
        coord_u = 0.
        coord_v = 0.
        sphere_flag = False
        min_depth = ti.select(min_depth > 0.0, min_depth - 1e-4, 1e7)
        for aabb_idx in range(self.num_objects):
            aabb_intersect, t_near, _f = self.aabb_test(aabb_idx, ray, start_p)
            if aabb_intersect == False: continue
            if t_near > min_depth: continue
            start_id  = self.obj_info[aabb_idx, 0]
            is_sphere = self.obj_info[aabb_idx, 2]
            if is_sphere:
                center  = self.prims[start_id, 0]
                radius2 = self.prims[start_id, 1][0] ** 2
                s2c     = center - start_p
                center_norm2 = s2c.norm_sqr()
                proj_norm = tm.dot(ray, s2c)
                c2ray_norm = center_norm2 - proj_norm ** 2  # center to ray distance ** 2
                if c2ray_norm >= radius2: continue
                ray_t = proj_norm
                ray_cut = ti.sqrt(radius2 - c2ray_norm)
                ray_t += ti.select(center_norm2 > radius2 + 1e-4, -ray_cut, ray_cut)
                if ray_t > 1e-4 and ray_t < min_depth:
                    min_depth = ray_t
                    obj_id = aabb_idx
                    prm_id = start_id
                    sphere_flag = True
            else:
                tri_num  = self.obj_info[aabb_idx, 1]
                for mesh_idx in range(start_id, tri_num + start_id):
                    p1 = self.prims[mesh_idx, 0]
                    v1 = self.precom_vec[mesh_idx, 0]
                    v2 = self.precom_vec[mesh_idx, 1]
                    mat = ti.Matrix.cols([v1, v2, -ray]).inverse()
                    u, v, t = mat @ (start_p - p1)
                    if u >= 0 and v >= 0 and u + v <= 1.0:
                        if t > 1e-4 and t < min_depth:
                            min_depth = t
                            obj_id = aabb_idx
                            prm_id = mesh_idx
                            coord_u = u
                            coord_v = v
                            sphere_flag = False
        n_g = vec3([1, 0, 0])
        n_s = vec3([1, 0, 0])
        if obj_id >= 0:
            if sphere_flag:
                center = self.prims[prm_id, 0]
                n_g = (start_p + min_depth * ray - center).normalized() 
                coord_u = (tm.atan2(n_g[1], n_g[0]) + tm.pi) * INV_2PI
                coord_v = tm.acos(n_g[2]) * INV_PI
                n_s = n_g
            else:
                n_g = self.normals[prm_id]
                # calculate vertex normal (for shading) if vertex normal exists
                if ti.static(self.has_v_normal):
                    n_s = self.v_normals[prm_id, 0] * (1. - coord_u - coord_v) + \
                        coord_u * self.v_normals[prm_id, 1] + \
                        coord_v * self.v_normals[prm_id, 2]
                else: 
                    n_s = n_g
        # We should calculate global uv and n_s outside
        return Interaction(
            obj_id = obj_id, prim_id = prm_id, n_g = n_g, n_s = n_s,
            uv = vec2(coord_u, coord_v), min_depth = min_depth
        )

    @ti.func
    def does_intersect(self, ray, start_p, min_depth = -1.0):
        """ Villina intersection test logic without acceleration structure """
        hit_flag = False
        min_depth = ti.select(min_depth > 0.0, min_depth - 1e-4, 1e7)
        for aabb_idx in range(self.num_objects):
            aabb_intersect, t_near, _f = self.aabb_test(aabb_idx, ray, start_p)
            if aabb_intersect == False: continue
            if t_near > min_depth: continue
            start_id  = self.obj_info[aabb_idx, 0]
            is_sphere = self.obj_info[aabb_idx, 2]
            if is_sphere:
                center  = self.prims[start_id, 0]
                radius2 = self.prims[start_id, 1][0] ** 2
                s2c     = center - start_p
                center_norm2 = s2c.norm_sqr()
                proj_norm = tm.dot(ray, s2c)
                c2ray_norm = center_norm2 - proj_norm ** 2  # center to ray distance ** 2
                if c2ray_norm >= radius2: continue
                ray_t = proj_norm
                if center_norm2 > radius2 + 1e-4:
                    ray_t -= ti.sqrt(radius2 - c2ray_norm)
                else:
                    ray_t += ti.sqrt(radius2 - c2ray_norm)
                if ray_t > 1e-4 and ray_t < min_depth:
                    hit_flag = True
            else:
                tri_num   = self.obj_info[aabb_idx, 1]
                for mesh_idx in range(start_id, tri_num + start_id):
                    p1 = self.prims[mesh_idx, 0]
                    v1 = self.precom_vec[mesh_idx, 0]
                    v2 = self.precom_vec[mesh_idx, 1]
                    mat = ti.Matrix.cols([v1, v2, -ray]).inverse()
                    u, v, t = mat @ (start_p - p1)
                    if u >= 0 and v >= 0 and u + v <= 1.0:
                        if t > 1e-4 and t < min_depth:
                            hit_flag = True
                            break
            if hit_flag: break
        return hit_flag

    @ti.kernel
    def render(self, t_start: int, t_end: int, s_start: int, s_end: int, max_bnc: int, max_depth: int):
        pass

    @ti.kernel
    def reset(self):
        pass
    
if __name__ == "__main__":
    ti.init()
    _, meshes, configs = scene_parsing("../scene/test/", "test.xml")
    base = TracerBase(meshes, configs)
