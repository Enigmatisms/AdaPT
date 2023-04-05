"""
    Tracer for direct lighting given a point source
    Blinn-Phong model
    @author: Qianyue He
    @date: 2023.1.22
"""

import sys
sys.path.append("..")

import taichi as ti
import taichi.math as tm
from taichi.math import vec3

from typing import List
from la.cam_transform import *
from emitters.point import PointSource
from tracer.tracer_base import TracerBase

from parser.obj_desc import ObjDescriptor
from parser.xml_parser import mitsuba_parsing

REDENDER_DEPTH = False

@ti.data_oriented
class BlinnPhongTracer(TracerBase):
    """
        Simple Ray tracing using Bary-centric coordinates
        The tracer can only trace direct components currently
        This tracer only supports one point source, since this is a simple one
        origin + direction * t = u * PA(vec) + v * PB(vec) + P
        This is a rank-3 matrix linear equation
    """
    def __init__(self, emitter: PointSource, objects: List[ObjDescriptor], prop: dict):
        super().__init__(objects, prop)
        self.emit_int = vec3(emitter.intensity, dt = float)   
            
        self.surf_color = ti.Vector.field(3, float, self.num_objects)
        self.shininess  = ti.Vector.field(3, float, self.num_objects)
        self.depth_map  = ti.field(float, (self.w, self.h))                # output: gray-scale

        self.initialze(objects)

    def initialze(self, objects: List[ObjDescriptor]):
        for i, obj in enumerate(objects):
            for j, (mesh, normal) in enumerate(zip(obj.meshes, obj.normals)):
                for k, vec in enumerate(mesh):
                    self.meshes[i, j, k]  = vec3(vec)
                if mesh.shape[0] > 2:       # not a sphere
                    self.precom_vec[i, j, 0] = self.meshes[i, j, 1] - self.meshes[i, j, 0]                    
                    self.precom_vec[i, j, 1] = self.meshes[i, j, 2] - self.meshes[i, j, 0]                     
                self.normals[i, j] = vec3(normal) 
            self.mesh_cnt[i]    = obj.tri_num
            self.shininess[i]   = obj.bsdf.k_g
            self.aabbs[i, 0]    = vec3(obj.aabb[0])       # unrolled
            self.aabbs[i, 1]    = vec3(obj.aabb[1])
            self.surf_color[i]  = vec3(obj.bsdf.k_d)

    @ti.kernel
    def render(self, emit_pos: vec3):
        for i, j in self.pixels:
            ray = self.pix2ray(i, j)
            obj_id, normal, min_depth = self.ray_intersect(ray, self.cam_t)
            # Iterate through all the meshes and find the minimum depth
            if obj_id >= 0:
                if ti.static(REDENDER_DEPTH):
                    self.depth_map[i, j] = min_depth
                # Calculate Blinn-Phong lighting model
                hit_point  = ray * min_depth + self.cam_t
                to_emitter = emit_pos - hit_point
                emitter_d  = to_emitter.norm()
                light_dir  = to_emitter / emitter_d
                # light_dir and ray are normalized, ray points from cam to hit point
                # the ray direction vector in half way vector should point from hit point to cam
                half_way = (0.5 * (light_dir - ray)).normalized()
                spec = tm.pow(ti.max(tm.dot(half_way, normal), 0.0), self.shininess[obj_id])
                spec *= ti.min(1.0 / (1e-5 + emitter_d ** 2), 1e5)
                if self.does_intersect(light_dir, hit_point, emitter_d):
                    spec *= 0.1
                self.pixels[i, j] = spec * self.emit_int * self.surf_color[obj_id]
            else:
                if ti.static(REDENDER_DEPTH):
                    self.depth_map[i, j] = 0.0
                self.pixels[i, j].fill(0.0)

if __name__ == "__main__":
    profiling = False
    ti.init(arch = ti.vulkan, kernel_profiler = profiling)
    emitter_configs, _, meshes, configs = mitsuba_parsing("../scene/test/", "point_cbox.xml")
    emitter = emitter_configs[0]
    emitter_pos = vec3(emitter.pos)
    bpt = BlinnPhongTracer(emitter, meshes, configs)
    # Note that direct test the rendering time (once) is meaningless, executing for the first time
    # will be accompanied by JIT compiling, compilation time will be included.
    gui = ti.GUI('BPT', (bpt.w, bpt.h))
    while gui.running:
        for e in gui.get_events(gui.PRESS):
            if e.key == gui.ESCAPE:
                gui.running = False
            elif e.key == 'a':
                emitter_pos[0] += 0.05
            elif e.key == 'd':
                emitter_pos[0] -= 0.05
            elif e.key == gui.DOWN:
                emitter_pos[1] -= 0.05
            elif e.key == gui.UP:
                emitter_pos[1] += 0.05
            elif e.key == 'd':
                emitter_pos[0] += 0.05
            elif e.key == 's':
                emitter_pos[2] -= 0.05
            elif e.key == 'w':
                emitter_pos[2] += 0.05
        bpt.render(emitter_pos)
        gui.set_image(bpt.pixels)
        gui.show()
        if gui.running == False: break
        gui.clear()
        bpt.reset()

    if profiling:
        ti.profiler.print_kernel_profiler_info() 
    ti.tools.imwrite(bpt.pixels.to_numpy(), "./blinn-phong.png")

    if REDENDER_DEPTH:
        depth_map = bpt.depth_map.to_numpy()
        depth_map /= depth_map.max()
        ti.tools.imwrite(depth_map, "./depth.png")
