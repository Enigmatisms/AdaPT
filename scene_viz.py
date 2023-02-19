"""
    Scene visualizer for FoV and pose setting
    @author: Qianyue He
    @date: 2023-2-18
"""
import os
import sys
sys.path.append("..")

import numpy as np
import taichi as ti
import taichi.math as tm
from taichi.math import vec3, mat3

from typing import List
from la.cam_transform import *

from scene.obj_desc import ObjDescriptor
from scene.xml_parser import mitsuba_parsing
from scene.opts import get_options
from utils.tools import folder_path

@ti.data_oriented
class Visualizer:
    """ Emitter is not supported. We only import mesh / sphere object in here """
    def __init__(self, objects: List[ObjDescriptor], prop: dict):
        self.w          = ti.field(ti.i32, ())
        self.h          = ti.field(ti.i32, ())
        self.focal      = ti.field(ti.f32, ())
        self.inv_focal  = ti.field(ti.f32, ())
        self.half_w     = ti.field(ti.f32, ())
        self.half_h     = ti.field(ti.f32, ())
        self.w[None]    = prop['film']['width']                              # image is a standard square
        self.h[None]    = prop['film']['height']

        self.focal[None]     = fov2focal(prop['fov'], min(self.w[None], self.h[None]))
        self.inv_focal[None] = 1. / self.focal[None]
        self.half_w[None]    = float(self.w[None]) / 2.
        self.half_h[None]    = float(self.h[None]) / 2.

        self.num_objects = len(objects)
        max_tri_num = max([obj.tri_num for obj in objects])

        self.cam_orient = ti.Vector.field(3, float, ())
        self.cam_t      = ti.Vector.field(3, float, ())
        self.cam_r      = ti.Matrix.field(3, 3, float, ())

        cam_orient            = prop['transform'][0]
        cam_orient            /= np.linalg.norm(cam_orient)
        self.cam_orient[None] = vec3(cam_orient)
        self.cam_t[None]      = vec3(prop['transform'][1])
        self.cam_r[None]      = mat3(np_rotation_between(np.float32([0, 0, 1]), cam_orient))
        
        self.aabbs      = ti.Vector.field(3, float, (self.num_objects, 2))
        self.normals    = ti.Vector.field(3, float)
        self.meshes     = ti.Vector.field(3, float)                    # leveraging SSDS, shape (N, mesh_num, 3) - vector3d
        self.precom_vec = ti.Vector.field(3, float)
        self.pixels     = ti.Vector.field(3, float, (1024, 1024))      # maximum size: 1024

        self.bitmasked_nodes = ti.root.dense(ti.i, self.num_objects).bitmasked(ti.j, max_tri_num)
        self.bitmasked_nodes.place(self.normals)
        self.bitmasked_nodes.bitmasked(ti.k, 3).place(self.meshes)      # for simple shapes, this would be efficient
        self.bitmasked_nodes.dense(ti.k, 3).place(self.precom_vec)
        self.mesh_cnt   = ti.field(int, self.num_objects)
        self.ray_field  = ti.Vector.field(3, float, (1024, 1024))
        self.initialze(objects)
        self.initialize_rays()

    def initialze(self, objects: List[ObjDescriptor]):
        for i, obj in enumerate(objects):
            for j, (mesh, normal) in enumerate(zip(obj.meshes, obj.normals)):
                self.normals[i, j] = vec3(normal) 
                for k, vec in enumerate(mesh):
                    self.meshes[i, j, k]  = vec3(vec)
                if mesh.shape[0] > 2:       # not a sphere
                    self.precom_vec[i, j, 0] = self.meshes[i, j, 1] - self.meshes[i, j, 0]                    
                    self.precom_vec[i, j, 1] = self.meshes[i, j, 2] - self.meshes[i, j, 0]             
                    self.precom_vec[i, j, 2] = self.meshes[i, j, 0]
                else:
                    self.precom_vec[i, j, 0] = self.meshes[i, j, 0]
                    self.precom_vec[i, j, 1] = self.meshes[i, j, 1]
            self.mesh_cnt[i]    = obj.tri_num
            self.aabbs[i, 0]    = vec3(obj.aabb[0])        # unrolled
            self.aabbs[i, 1]    = vec3(obj.aabb[1])

    @ti.kernel
    def initialize_rays(self):
        inv_focal = self.inv_focal[None]
        # This is some what trivial
        cam_r = self.cam_r[None]
        for i, j in self.ray_field:
            cam_dir = vec3([(512. + 0.5 - float(i)) * inv_focal, (float(j) - 512. + 0.5) * inv_focal, 1.])
            self.ray_field[i, j] = (cam_r @ cam_dir).normalized()

    @ti.func
    def aabb_test(self, aabb_idx, ray: vec3, ray_o: vec3):
        """ AABB used to skip some of the objects """
        t_min = (self.aabbs[aabb_idx, 0] - ray_o) / ray
        t_max = (self.aabbs[aabb_idx, 1] - ray_o) / ray
        t1 = ti.min(t_min, t_max)
        t2 = ti.max(t_min, t_max)
        t_near  = ti.max(ti.max(t1.x, t1.y), t1.z)
        t_far   = ti.min(ti.min(t2.x, t2.y), t2.z)
        return t_near < t_far

    @ti.func
    def ray_intersect(self, ray, start_p):
        obj_id = -1
        tri_id = -1
        min_depth = 1e7
        for aabb_idx in range(self.num_objects):
            if self.aabb_test(aabb_idx, ray, start_p) == False: continue
            tri_num = self.mesh_cnt[aabb_idx]
            if tri_num:
                for mesh_idx in range(tri_num):
                    normal = self.normals[aabb_idx, mesh_idx]   # back-face culling removed
                    p1 = self.meshes[aabb_idx, mesh_idx, 0]
                    vec1 = self.precom_vec[aabb_idx, mesh_idx, 0]
                    vec2 = self.precom_vec[aabb_idx, mesh_idx, 1]
                    mat = ti.Matrix.cols([vec1, vec2, -ray]).inverse()
                    u, v, t = mat @ (start_p - p1)
                    if u >= 0 and v >= 0 and u + v <= 1.0:
                        if t > 5e-4 and t < min_depth:
                            min_depth = t
                            obj_id = aabb_idx
                            tri_id = mesh_idx
            else:
                center  = self.meshes[aabb_idx, 0, 0]
                radius2 = self.meshes[aabb_idx, 0, 1][0] ** 2
                s2c     = center - start_p
                center_norm2 = s2c.norm_sqr()
                proj_norm = tm.dot(ray, s2c)
                c2ray_norm = center_norm2 - proj_norm ** 2  # center to ray distance ** 2
                if c2ray_norm >= radius2: continue
                ray_t = proj_norm
                if center_norm2 > radius2 + 5e-4:
                    ray_t -= ti.sqrt(radius2 - c2ray_norm)
                else:
                    ray_t += ti.sqrt(radius2 - c2ray_norm)
                if ray_t > 5e-4 and ray_t < min_depth:
                    min_depth = ray_t
                    obj_id = aabb_idx
                    tri_id = -1
        normal = vec3([1, 0, 0])
        if obj_id >= 0:
            if tri_id < 0:
                center = self.meshes[obj_id, 0, 0]
                normal = (start_p + min_depth * ray - center).normalized() 
            else:
                normal = self.normals[obj_id, tri_id]
        return (obj_id, normal)

    @ti.kernel
    def render(self):
        for i, j in self.pixels:
            ray = self.ray_field[i, j]
            obj_id, normal = self.ray_intersect(ray, self.cam_t[None])
            if obj_id >= 0:
                self.pixels[i, j].fill(ti.max(-tm.dot(ray, normal), 0.))
            else:
                self.pixels[i, j].fill(0.0)

if __name__ == "__main__":
    options = get_options()
    cache_path = folder_path(f"./cached/viz/{options.scene}", f"Cache path for scene {options.scene} not found. JIT compiling...")
    ti.init(arch = ti.vulkan, default_ip = ti.i32, default_fp = ti.f32, offline_cache_file_path = cache_path)
    input_folder = os.path.join(options.input_path, options.scene)
    _, _, meshes, configs = mitsuba_parsing(input_folder, options.name)  # complex_cornell

    bpt = Visualizer(meshes, configs)
    gui = ti.GUI('Scene Interactive Visualizer', (1024, 1024))
    while gui.running:
        for e in gui.get_events(gui.PRESS):
            if e.key == gui.ESCAPE:
                gui.running = False
           
            # elif e.key == 'w':
                # emitter_pos[2] += 0.05
        bpt.render()
        gui.set_image(bpt.pixels)
        gui.show()
        if gui.running == False: break
        gui.clear()
