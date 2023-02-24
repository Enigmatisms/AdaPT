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

from scene.obj_desc import ObjDescriptor
from renderer.vpt import VolumeRenderer
from renderer.path_utils import Vertex
from renderer.constants import *

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
    def interact_mode(is_mi: int, hit_light: int):
        return ti.select(is_mi, VERTEX_MEDIUM, ti.select(hit_light >= 0, VERTEX_EMITTER, VERTEX_SURFACE))
    
    @staticmethod
    @ti.func
    def convert_density(pdf: float, this_pos: vec3, next_nv: vec3, next_pos: vec3, next_mi:int):
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
    def random_walk(self, ray_o, ray_d, normal, pdf: float, transport_mode: int):
        """ Random walk to generate path 
            pdf: initial pdf for this path
            transport mode: whether it is radiance or importance, 0 is camera radiance, 1 is light importance
            Before the random walk, corresponding initial vertex should be appended already
            TODO: logic check
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
            is_delta = (not is_mi) and self.is_delta(obj_id)
            vertex_args = {"_type": BDPT.interact_mode(is_mi, hit_light), "obj_id": obj_id, "emit_id": hit_light, 
                "bool_bits": in_free_space << 1 + is_delta, "pdf_fwd": pdf_fwd, "time": acc_time, 
                "normal": ZERO_V3 if is_mi else normal, "pos": hit_point, "ray_in": ray_d, "beta": throughput                
            }
            if transport_mode == TRANSPORT_RAD:         # Camera path
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
            ray_d, indirect_spec, ray_pdf = self.sample_new_ray(obj_id, old_ray_d, normal, is_mi, in_free_space, transport_mode)
            ray_o = hit_point
            throughput *= (indirect_spec / ray_pdf)

            # Step 6: re-evaluate backward PDF
            pdf_bwd = ray_pdf
            if not is_mi:   # If current sampling exists on the surface
                pdf_bwd = self.surface_pdf(obj_id, -old_ray_d, normal, -ray_d)
            if is_delta:
                pdf_fwd = pdf_bwd = 0.
            # ray_o is the position of the current vertex, which is used in prev vertex pdf_bwd
            if transport_mode == TRANSPORT_RAD:         # Camera transport mode
                self.cam_paths[prev_vid].set_bwd_pdf(pdf_bwd, ray_o)
            else:
                self.light_paths[prev_vid].set_bwd_pdf(pdf_bwd, ray_o)
        return vertex_num
    
    @ti.func
    def rasterize_pinhole(self, end_point):
        """ For path with only one camera vertex, ray should be re-rasterized to the film"""
        pass
    
    @ti.func
    def connect_path(self, sid: int, tid: int):
        le = ZERO_V3
        if sid == 0:            # light path is not used  
            vertex = self.cam_paths[tid - 1]
            if vertex._type == 2:   # is light?
                self.src_field[vertex.emit_id].eval_le(vertex.ray_in, vertex.normal)
        elif tid == 1:          # re-rasterize point onto the film, atomic add is allowed
            # need to track ray from light end point to camera
            vertex = self.light_paths[sid - 1]
            if vertex.is_connectible():
                ray_d = self.cam_t - vertex.pos
                depth = ray_d.norm()
                ray_d /= depth
                tr2cam = self.track_ray(ray_d, vertex.pos, depth)       # calculate transmittance from vertex to camera
                is_in_free_space = vertex.is_in_free_space()
                fr2cam = self.eval(vertex.obj_id, vertex.ray_in, ray_d, vertex.normal, vertex.is_mi, is_in_free_space)
                # divided by PDF
                pdf2cam = self.get_pdf(vertex.obj_id, vertex.ray_in, ray_d, vertex.normal, vertex.is_mi, is_in_free_space)
                fr2cam = ti.select(pdf2cam > 1e-5, fr2cam / pdf2cam, ZERO_V3)       # zero PDF returns zero contribution
                # light_path beta already carries L_e value
                tr_light = vertex.beta
                le = tr_light * tr2cam * fr2cam
        elif sid == 1:          # only one light vertex is used, resample
            vertex = self.cam_paths[tid - 1]
            if vertex.is_connectible():
                # randomly sample an emitter and corresponding point (direct component)
                emitter, emitter_pdf, emitter_valid = self.sample_light(vertex.emit_id)
                light_dir = vec3([0, 0, 0])
                # direct / emission component evaluation
                if emitter_valid:
                    emit_pos, shadow_int, direct_pdf = emitter.         \
                        sample(self.precom_vec, self.normals, self.mesh_cnt, vertex.pos)        # sample light
                # TODO: This is a little bit trickier. If a newly sampled vertex is created, we should use it in MIS
        else:                   # general cases
            pass
        pass

    @ti.func
    def bdpt_mis_weight():
        pass

    @ti.kernel
    def render():
        # Generate two path and connect them
        pass
