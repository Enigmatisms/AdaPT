"""
    Bidirectional Path Tracer
    Bidirectional volumetric path tracing method 
    @author: Qianyue He
    @date: 2023-2-20
"""


import taichi as ti
import taichi.math as tm
from taichi.math import vec3

from typing import List
from la.cam_transform import *
from emitters.abtract_source import LightSource

from bxdf.medium import Medium
from scene.obj_desc import ObjDescriptor
from sampler.general_sampling import balance_heuristic
from renderer.vpt import VolumeRenderer
from renderer.path_utils import Vertex

ZERO_V3 = vec3([0, 0, 0])

@ti.data_oriented
class BDPT(VolumeRenderer):
    def __init__(self, emitters: List[LightSource], objects: List[ObjDescriptor], prop: dict):
        super().__init__(emitters, objects, prop)
        
        self.light_paths = Vertex.field()       # (W, H, max_bounce + 1)
        self.cam_paths   = Vertex.field()       # (W, H, max_bounce + 2)

        self.path_nodes = ti.root.dense(ti.ij, (self.w, self.h))
        # light vertex is not included, therefore +1
        self.path_nodes.bitmasked(ti.k, self.max_bounce + 1).place(self.light_paths)       
        # camera vertex and extra light vertex is not included, therefore + 2
        self.path_nodes.bitmasked(ti.k, self.max_bounce + 2).place(self.cam_paths)        
        # Put "path generation", "path connection" and "MIS weight" in one function

    @staticmethod
    @ti.func
    def interact_mode(is_mi: ti.i32, hit_light: ti.i32):
        return ti.select(is_mi, 1, ti.select(hit_light >= 0, 2, 0))
    
    @staticmethod
    @ti.func
    def convert_density(pdf: ti.f32, this_pos: vec3, next_nv: vec3, next_pos: vec3, next_mi:ti.i32):
        """ Convert solid angle density to unit area density
            next_nv, next_pos, next_mi: normal vector / position / is_mi for the next vertex
        """
        diff_vec = next_pos - this_pos
        inv_norm2 = 1. / diff_vec.norm_sqr()
        pdf *= inv_norm2
        if not next_mi:
            pdf *= ti.abs(tm.dot(next_nv, diff_vec * ti.sqrt(inv_norm2)))
        return pdf

    @ti.func
    def random_walk(self, ray_o, ray_d, normal, pdf: ti.f32, transport_mode: ti.i32):
        """ Random walk to generate path 
            pdf: initial pdf for this path
            transport mode: whether it is radiance or importance
            Before the random walk, corresponding initial vertex should be appended already
        """
        old_pos    = ray_o
        normal     = vec3([0, 1, 0])
        throughput = vec3([1, 1, 1])
        vertex_num = 1
        acc_time   = 0.         # accumulated time
        ray_pdf    = pdf        # TODO: check whether the measure is correct

        while True:
            # Step 1: ray intersection
            obj_id, normal, min_depth = self.ray_intersect(ray_d, ray_o)

            if obj_id < 0: break                                # nothing is hit, break
            in_free_space = tm.dot(normal, ray_d) < 0
            # Step 2: check for mean free path sampling
            # Calculate mfp, path_beta = transmittance / PDF
            is_mi, min_depth, path_beta = self.sample_mfp(obj_id, in_free_space, min_depth) 
            hit_point = ray_d * min_depth + ray_o
            hit_light = -1 if is_mi else self.emitter_id[obj_id]
            acc_time += min_depth
            throughput *= path_beta         # attenuate first

            # Step 3: Create a new vertex and calculate pdf_fwd
            pdf_fwd = BDPT.convert_density(ray_pdf, old_pos, normal, hit_point, is_mi)
            vertex_args = {"_type": BDPT.interact_mode(is_mi, hit_light), "obj_id": obj_id, "emit_id": hit_light, 
                "is_delta": (not is_mi) and self.is_delta(obj_id), "pdf_fwd": pdf_fwd, "time": acc_time, 
                "normal": ZERO_V3 if is_mi else normal, "pos": hit_point, "ray_in": -ray_d, "beta": throughput                
            }
            if not transport_mode:         # Camera path transport_mode = 0
                self.cam_paths[vertex_num] = Vertex(**vertex_args) 
            else:                          # Light path
                self.light_paths[vertex_num] = Vertex(**vertex_args) 

            # Step 4: ray termination test - RR termination and max bounce. If ray terminates, we won't have to sample
            if vertex_num >= self.max_bounce:
                break
            max_value = throughput.max()
            if ti.random(float) > max_value: break
            else: throughput *= 1. / ti.max(max_value, 1e-7)    # unbiased calculation
            prev_vid = vertex_num - 1
            vertex_num += 1

            # Step 5: sample new ray. This should distinguish between surface and medium interactions
            old_ray_d = ray_d
            ray_d, indirect_spec, ray_pdf = self.sample_new_ray(obj_id, old_ray_d, normal, is_mi, in_free_space)
            ray_o = hit_point
            throughput *= (indirect_spec / ray_pdf)

            # Step 6: re-evaluate backward PDF
            pdf_bwd = ray_pdf
            if not is_mi:   # If current sampling exists on the surface
                pdf_bwd = self.get_pdf(obj_id, -old_ray_d, normal, -ray_d)
            # ray_o is the position of the current vertex, which is used in prev vertex pdf_bwd
            if not transport_mode:         # Camera transport mode
                self.cam_paths[prev_vid].set_bwd_pdf(pdf_bwd, ray_o)
            else:
                self.light_paths[prev_vid].set_bwd_pdf(pdf_bwd, ray_o)
        return vertex_num
    
    @ti.func
    def connect_paths():
        pass

    @ti.func
    def bdpt_mis_weight():
        pass

    @ti.kernel
    def render():
        # Generate two path and connect them
        pass
