"""
    Vanilla renderer without volumetric scattering
    @author: Qianyue He
    @date: 2023.2.7
"""
import taichi as ti
from taichi.math import vec3

from typing import List
from la.cam_transform import *
from tracer.path_tracer import PathTracer
from emitters.abtract_source import LightSource

from scene.obj_desc import ObjDescriptor
from sampler.general_sampling import balance_heuristic

@ti.data_oriented
class Renderer(PathTracer):
    """
        Renderer Final Class
    """
    def __init__(self, emitters: List[LightSource], objects: List[ObjDescriptor], prop: dict):
        super().__init__(emitters, objects, prop)
        
    @ti.kernel
    def render(self):
        self.cnt[None] += 1
        for i, j in self.pixels:
            ray_d = self.pix2ray(i, j)
            ray_o = self.cam_t
            obj_id, normal, min_depth = self.ray_intersect(ray_d, ray_o)
            hit_light       = self.emitter_id[obj_id]   # id for hit emitter, if nothing is hit, this value will be -1
            color           = vec3([0, 0, 0])
            contribution    = vec3([1, 1, 1])
            emission_weight = 1.0
            for _i in range(self.max_bounce):
                if obj_id < 0: break                    # nothing is hit, break
                if ti.static(self.use_rr):
                    # Simple Russian Roullete ray termination
                    max_value = contribution.max()
                    if ti.random(float) > max_value: break
                    else: contribution *= 1. / (max_value + 1e-7)    # unbiased calculation
                else:
                    if contribution.max() < 1e-4: break     # contribution too small, break
                hit_point   = ray_d * min_depth + ray_o

                direct_pdf  = 1.0
                emitter_pdf = 1.0
                break_flag  = False
                shadow_int  = vec3([0, 0, 0])
                direct_int  = vec3([0, 0, 0])
                direct_spec = vec3([1, 1, 1])
                for _j in range(self.num_shadow_ray):    # more shadow ray samples
                    emitter, emitter_pdf, emitter_valid = self.sample_light(hit_light)
                    light_dir = vec3([0, 0, 0])
                    # direct / emission component evaluation
                    if emitter_valid:
                        emit_pos, shadow_int, direct_pdf, _n = emitter.         \
                            sample(self.precom_vec, self.normals, self.mesh_cnt, hit_point)        # sample light
                        to_emitter  = emit_pos - hit_point
                        emitter_d   = to_emitter.norm()
                        light_dir   = to_emitter / emitter_d
                        # Note that, null surface in vanilla renderer will produce erroneous results
                        if self.does_intersect(light_dir, hit_point, emitter_d):        # shadow ray 
                            shadow_int.fill(0.0)
                        else:
                            direct_spec = self.eval(obj_id, ray_d, light_dir, normal, False, False)
                    else:       # the only situation for being invalid, is when there is only one source and the ray hit the source
                        break_flag = True
                        break
                    light_pdf = emitter_pdf * direct_pdf
                    if ti.static(self.use_mis):
                        mis_w = 1.0
                        if not emitter.is_delta_source():
                            bsdf_pdf = self.surface_pdf(obj_id, light_dir, normal, ray_d)
                            mis_w    = balance_heuristic(light_pdf, bsdf_pdf)
                        direct_int  += direct_spec * shadow_int * mis_w / emitter_pdf
                    else:
                        direct_int += direct_spec * shadow_int / emitter_pdf
                if not break_flag:
                    direct_int *= self.inv_num_shadow_ray
                # emission: ray hitting an area light source
                emit_int    = vec3([0, 0, 0])
                if hit_light >= 0:
                    emit_int = self.src_field[hit_light].eval_le(hit_point - ray_o, normal)
                
                # indirect component requires sampling 
                ray_d, indirect_spec, ray_pdf = self.sample_new_ray(obj_id, ray_d, normal, False, False)
                ray_o = hit_point
                color += (direct_int + emit_int * emission_weight) * contribution
                # VERY IMPORTANT: rendering should be done according to rendering equation (approximation)
                contribution *= indirect_spec / ray_pdf
                obj_id, normal, min_depth = self.ray_intersect(ray_d, ray_o)

                if obj_id >= 0:
                    hit_light = self.emitter_id[obj_id]
                    if ti.static(self.use_mis):
                        emitter_pdf = 0.0
                        if hit_light >= 0 and self.is_delta(obj_id) == 0:
                            emitter_pdf = self.src_field[hit_light].solid_angle_pdf(ray_d, normal, min_depth)
                        emission_weight = balance_heuristic(ray_pdf, emitter_pdf)

            self.color[i, j] += ti.select(ti.math.isnan(color), 0., color)
            self.pixels[i, j] = self.color[i, j] / self.cnt[None]
    
    def summary(self):
        print(f"[INFO] PT Finished rendering. SPP = {self.cnt[None]}. Rendering time: {self.clock.toc():.3f} s")
