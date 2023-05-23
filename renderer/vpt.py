"""
    Volumetric Path Tracer
    Participating media is bounded by object or world AABB
    @author: Qianyue He
    @date: 2023-2-13
"""

import taichi as ti
import taichi.math as tm
from taichi.math import vec3

from typing import List
from la.cam_transform import *
from tracer.path_tracer import PathTracer
from emitters.abtract_source import LightSource

from parsers.obj_desc import ObjDescriptor

from rich.console import Console
CONSOLE = Console(width = 128)

"""
   MIS is required to make convergence faster and supress variance
"""

@ti.data_oriented
class VolumeRenderer(PathTracer):
    """
        Volumetric Renderer Class
    """
    def __init__(self, emitters: List[LightSource], objects: List[ObjDescriptor], prop: dict):
        super().__init__(emitters, objects, prop)
        self.world_scattering = self.world.medium._type >= 0
        
    @ti.func
    def get_transmittance(self, idx: int, in_free_space: int, depth: float):
        transmittance = vec3([1., 1., 1.])
        world_valid_scat = in_free_space and self.world_scattering
        if world_valid_scat or self.is_scattering(idx):
            if world_valid_scat:
                transmittance = self.world.medium.transmittance(depth)
            elif not in_free_space:
                # if not in_free_space, bsdf_field[idx] must be valid
                transmittance = self.bsdf_field[idx].medium.transmittance(depth)
        return transmittance
    
    @ti.func
    def non_null_surface(self, idx: int):
        non_null = True
        # All these idx >= 0 check is for world scattering medium
        if idx >= 0 and not ti.is_active(self.brdf_nodes, idx):      # BRDF is non-null, BSDF can be non-null
            non_null = self.bsdf_field[idx].is_non_null()
        return non_null

    @ti.func
    def sample_mfp(self, idx: int, in_free_space: int, depth: float):
        """ Mean free path sampling, returns: 
            - whether medium is a valid scattering medium / mean free path
        """
        is_mi = False
        mfp   = depth
        beta  = vec3([1., 1., 1.])
        # whether the world is valid for scattering: inside the free space and world has scattering medium
        world_valid_scat = in_free_space and self.world_scattering      
        if world_valid_scat or self.is_scattering(idx):
            # Note that the if / else order is not interchangable, since it is possible that
            # world_valid_scat = True and self.is_scattering(idx) = True, in this situation
            # world scattering should be evaluated
            if world_valid_scat:        # scattering is not in the free space
                is_mi, mfp, beta = self.world.medium.sample_mfp(depth)
            elif not in_free_space:
                is_mi, mfp, beta = self.bsdf_field[idx].medium.sample_mfp(depth)
            # use medium to sample / calculate transmittance
        return is_mi, mfp, beta
    
    @ti.func
    def track_ray(self, ray, start_p, depth):
        """ 
            For medium interaction, check if the path to one point is not blocked (by non-null surface object)
            And also we need to calculate the attenuation along the path, e.g.: if the ray passes through
            two clouds of smoke and between the two clouds there is a transparent medium
            FIXME: the speed of this method should be boosted
        """
        tr = vec3([1., 1., 1.])
        in_free_space = True
        cur_point = start_p
        cur_ray = ray.normalized()
        acc_depth = 0.0
        for _i in range(7):             # maximum tracking depth = 7 (corresponding to at most 2 clouds of smoke)
            obj_id, normal, min_depth, _p, _u, _v = self.ray_intersect(cur_ray, cur_point, depth)
            if obj_id < 0:     
                acc_depth += depth * self.world.medium.ior      # definitely not in an object
                if not self.world_scattering: break             # nothing is hit, break
                else:                                           # the world is filled with scattering medium
                    min_depth = depth
                    in_free_space = True
                    obj_id = -1
            else:
                if self.non_null_surface(obj_id):               # non-null surface blocks the ray path, break
                    tr.fill(0.0)                                # travelling time need not to be calculated here since we will abandon this path
                    break  
                in_free_space = tm.dot(normal, cur_ray) < 0
                acc_depth += min_depth * self.get_ior(obj_id, in_free_space)
            # invalid medium can be "BRDF" or "transparent medium". Transparent medium has non-null surface, therefore invalid
            # TODO: I feel something is not right here...
            transmittance = self.get_transmittance(obj_id, in_free_space, min_depth)
            tr *= transmittance
            cur_point += cur_ray * min_depth
            depth -= min_depth
            if depth <= 5e-5: break     # reach the target point: break
        return tr, acc_depth
    
    @ti.func
    def world_bound_time(self, ray_o, ray_d):
        t_min = (self.w_aabb_min - ray_o) / ray_d
        t_max = (self.w_aabb_max - ray_o) / ray_d
        return ti.max(t_min, t_max).min()
        
    @ti.kernel
    def render(self, _t_start: int, _t_end: int, _s_start: int, _s_end: int, _a: int, _b: int):
        self.cnt[None] += 1
        for i, j in self.pixels:
            in_crop_range = i >= self.start_x and i < self.end_x and j >= self.start_y and j < self.end_y
            if not self.do_crop or in_crop_range:
                # TODO: MIS in VPT is not considered yet (too complex)
                ray_d = self.pix2ray(i, j)
                ray_o = self.cam_t
                normal          = vec3([0, 1, 0])
                color           = vec3([0, 0, 0])
                throughput      = vec3([1, 1, 1])
                in_free_space = True
                bounce = 0
                while True:
                    # for _i in range(self.max_bounce):
                    # Step 1: ray termination test - Only RR termination is allowed
                    max_value = throughput.max()
                    if ti.random(float) > max_value: break
                    else: throughput *= 1. / ti.max(max_value, 1e-7)    # unbiased calculation
                    # Step 2: ray intersection
                    obj_id, normal, min_depth, prim_id, u_coord, v_coord = self.ray_intersect(ray_d, ray_o)
                    if obj_id < 0:     
                        if not self.world_scattering: break     # nothing is hit, break
                        else:                                   # the world is filled with scattering medium
                            min_depth = self.world_bound_time(ray_o, ray_d)
                            in_free_space = True
                            obj_id = -1
                    else:
                        in_free_space = tm.dot(normal, ray_d) < 0
                    # Step 3: check for mean free path sampling
                    # Calculate mfp, path_beta = transmittance / PDF
                    is_mi, min_depth, path_beta = self.sample_mfp(obj_id, in_free_space, min_depth) 
                    if obj_id < 0 and not is_mi: break          # exiting world bound
                    hit_point = ray_d * min_depth + ray_o
                    throughput *= path_beta         # attenuate first
                    if not is_mi and not self.non_null_surface(obj_id):
                        ray_o = hit_point
                        continue
                    hit_light = -1 if is_mi else self.emitter_id[obj_id]
                    # Step 4: direct component estimation
                    emitter_pdf = 1.0
                    break_flag  = False
                    shadow_int  = vec3([0, 0, 0])
                    direct_int  = vec3([0, 0, 0])
                    direct_spec = vec3([1, 1, 1])
                    tex = self.get_uv_color(obj_id, prim_id, u_coord, v_coord)
                    for _j in range(self.num_shadow_ray):    # more shadow ray samples
                        emitter, emitter_pdf, emitter_valid, _ei = self.sample_light(hit_light)
                        light_dir = vec3([0, 0, 0])
                        # direct / emission component evaluation
                        if emitter_valid:
                            emit_pos, shadow_int, _d, _n = emitter.         \
                                sample_hit(self.precom_vec, self.normals, self.obj_info, hit_point)        # sample light
                            to_emitter  = emit_pos - hit_point
                            emitter_d   = to_emitter.norm()
                            light_dir   = to_emitter / emitter_d
                            tr, _ = self.track_ray(light_dir, hit_point, emitter_d)
                            shadow_int *= tr
                            direct_spec = self.eval(obj_id, ray_d, light_dir, normal, is_mi, in_free_space, tex = tex)
                        else:       # the only situation for being invalid, is when there is only one source and the ray hit the source
                            break_flag = True
                            break
                        direct_int += direct_spec * shadow_int / emitter_pdf
                    if not break_flag:
                        direct_int *= self.inv_num_shadow_ray
                    # Step 5: emission evaluation - ray hitting an area light source
                    emit_int    = vec3([0, 0, 0])
                    if hit_light >= 0:
                        emit_int = self.src_field[hit_light].eval_le(hit_point - ray_o, normal)

                    # Step 6: sample new ray. This should distinguish between surface and medium interactions
                    ray_d, indirect_spec, ray_pdf = self.sample_new_ray(obj_id, ray_d, normal, is_mi, in_free_space, tex = tex)
                    ray_o = hit_point
                    color += (direct_int + emit_int) * throughput
                    if not is_mi:
                        if indirect_spec.max() == 0. or ray_pdf == 0.: break
                        throughput *= (indirect_spec / ray_pdf)
                    bounce += 1
                    if bounce >= self.max_bounce:
                        break

                self.color[i, j] += ti.select(ti.math.isnan(color), 0., color)
                self.pixels[i, j] = self.color[i, j] / self.cnt[None]
            
    def summary(self):
        super().summary()
        CONSOLE.print(f"VPT SPP = {self.cnt[None]}. Rendering time: {self.clock.toc():.3f} s", justify="center")
        