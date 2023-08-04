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
from sampler.general_sampling import balance_heuristic

from rich.console import Console
CONSOLE = Console(width = 128)

"""
   MIS is required to make convergence faster and supress variance
"""

@ti.data_oriented
class VolumeRenderer(PathTracer):
    """ Volumetric Renderer Class
    """
    def __init__(self, 
        emitters: List[LightSource], array_info: dict, bxdfs,
        objects: List[ObjDescriptor], prop: dict, bvh_delay: bool = False
    ):
        super().__init__(emitters, array_info, bxdfs, objects, prop, bvh_delay)
        self.world_scattering = self.world.medium._type >= 0
        
    @ti.func
    def get_transmittance(self, idx: int, in_free_space: int, depth: float):
        transmittance = vec3([1., 1., 1.])
        world_valid_scat = in_free_space and self.world_scattering
        if world_valid_scat or self.is_scattering(idx):
            if world_valid_scat:
                transmittance = self.world.medium.transmittance(depth)
            elif not in_free_space:
                # if not in_free_space, bsdf_idx must be valid
                bsdf_idx = self.mix_field[idx].comps[3]
                transmittance = self.bsdf_field[bsdf_idx].medium.transmittance(depth)
        return transmittance
    
    @ti.func
    def non_null_surface(self, idx: int):
        non_null = True
        # All these idx >= 0 check is for world scattering medium
        if idx >= 0:
            bsdf_idx = self.mix_field[idx].non_null_index()
            if bsdf_idx >= 0 and not ti.is_active(self.bxdf_nodes, bsdf_idx):      # BRDF is non-null, BSDF can be non-null
                non_null = self.bsdf_field[bsdf_idx].is_non_null()
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
                bsdf_index = self.mix_field[idx].comps[3]
                is_mi, mfp, beta = self.bsdf_field[bsdf_index].medium.sample_mfp(depth)
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
        cur_ray = ray
        acc_depth = 0.0
        for _i in range(7):             # maximum tracking depth = 7 (corresponding to at most 2 clouds of smoke)
            it = self.ray_intersect(cur_ray, cur_point, depth)
            if it.obj_id < 0:     
                acc_depth += depth * self.world.medium.ior      # definitely not in an object
                if not self.world_scattering: break             # nothing is hit, break
                else:                                           # the world is filled with scattering medium
                    it.min_depth = depth
                    in_free_space = True
                    it.obj_id = -1
            else:
                if self.non_null_surface(it.obj_id):               # non-null surface blocks the ray path, break
                    tr.fill(0.0)                                # travelling time need not to be calculated here since we will abandon this path
                    break  
                in_free_space = tm.dot(it.n_g, cur_ray) < 0
                acc_depth += it.min_depth * self.get_ior(it.obj_id, in_free_space)
            # invalid medium can be "BRDF" or "transparent medium". Transparent medium has non-null surface, therefore invalid
            # TODO: I feel something is not right here...
            transmittance = self.get_transmittance(it.obj_id, in_free_space, it.min_depth)
            tr *= transmittance
            cur_point += cur_ray * it.min_depth
            depth -= it.min_depth
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
        ti.loop_config(parallelize = 8, block_dim = 512)
        for i, j in self.pixels:
            in_crop_range = i >= self.start_x and i < self.end_x and j >= self.start_y and j < self.end_y
            if not self.do_crop or in_crop_range:
                # TODO: MIS in VPT (now I have a good understanding for VPT) 
                ray_d = self.pix2ray(i, j)
                ray_o = self.cam_t
                color           = vec3([0, 0, 0])
                throughput      = vec3([1, 1, 1])
                emission_weight = 1.0

                in_free_space = True
                bounce = 0
                while True:
                    # for _i in range(self.max_bounce):
                    # Step 1: ray termination test - Only RR termination is allowed
                    max_value = throughput.max()
                    if ti.random(float) > max_value: break
                    else: throughput *= 1. / ti.max(max_value, 1e-7)    # unbiased calculation
                    # Step 2: ray intersection
                    it = self.ray_intersect(ray_d, ray_o)
                    if it.obj_id < 0:     
                        if not self.world_scattering: break     # nothing is hit, break
                        else:                                   # the world is filled with scattering medium
                            it.min_depth = self.world_bound_time(ray_o, ray_d)
                            in_free_space = True
                            it.obj_id = -1
                    else:
                        in_free_space = tm.dot(it.n_g, ray_d) < 0
                    # Step 3: check for mean free path sampling
                    # Calculate mfp, path_beta = transmittance / PDF
                    is_mi, it.min_depth, path_beta = self.sample_mfp(it.obj_id, in_free_space, it.min_depth) 
                    if it.obj_id < 0 and not is_mi: break          # exiting world bound
                    hit_point = ray_d * it.min_depth + ray_o
                    throughput *= path_beta         # attenuate first
                    if not is_mi and not self.non_null_surface(it.obj_id):
                        ray_o = hit_point
                        continue
                    hit_light = -1 if is_mi else self.emitter_id[it.obj_id]
                    # Step 4: direct component estimation
                    emitter_pdf = 1.0
                    break_flag  = False
                    shadow_int  = vec3([0, 0, 0])
                    direct_int  = vec3([0, 0, 0])
                    direct_spec = vec3([1, 1, 1])
                    direct_pdf  = 1.
                    it.tex, _vl = self.get_uv_item(self.albedo_map, self.albedo_img, it)
                    for _j in range(self.num_shadow_ray):    # more shadow ray samples
                        emitter, emitter_pdf, emitter_valid, _ei = self.sample_light(hit_light)
                        light_dir = vec3([0, 0, 0])
                        # direct / emission component evaluation
                        if emitter_valid:
                            emit_pos, shadow_int, direct_pdf, _n = emitter.         \
                                sample_hit(self.precom_vec, self.normals, self.obj_info, hit_point)        # sample light
                            to_emitter  = emit_pos - hit_point
                            emitter_d   = to_emitter.norm()
                            light_dir   = to_emitter / emitter_d
                            tr, _ = self.track_ray(light_dir, hit_point, emitter_d)
                            shadow_int *= tr
                            direct_spec = self.eval(it, ray_d, light_dir, is_mi, in_free_space)
                        else:       # the only situation for being invalid, is when there is only one source and the ray hit the source
                            break_flag = True
                            break
                        light_pdf = emitter_pdf * direct_pdf
                        if ti.static(self.use_mis):                 # MIS for vpt
                            mis_w = 1.0
                            if not emitter.is_delta_pos():
                                bsdf_pdf = 1.
                                if is_mi:
                                    bsdf_pdf = direct_spec[0]       # all elements are phase function value 
                                else:
                                    bsdf_pdf = self.surface_pdf(it, light_dir, ray_d)
                                mis_w    = balance_heuristic(light_pdf, bsdf_pdf)
                            direct_int  += direct_spec * shadow_int * mis_w / emitter_pdf
                        else:
                            direct_int += direct_spec * shadow_int / emitter_pdf

                    if not break_flag:
                        direct_int *= self.inv_num_shadow_ray
                    # Step 5: emission evaluation - ray hitting an area light source
                    emit_int    = vec3([0, 0, 0])
                    if hit_light >= 0:
                        emit_int = self.src_field[hit_light].eval_le(hit_point - ray_o, it.n_g)

                    # Step 6: sample new ray. This should distinguish between surface and medium interactions
                    ray_d, indirect_spec, ray_pdf = self.sample_new_ray(it, ray_d, is_mi, in_free_space)
                    ray_o = hit_point
                    color += (direct_int + emit_int * emission_weight) * throughput
                    if not is_mi:
                        if indirect_spec.max() == 0. or ray_pdf == 0.: break
                        throughput *= (indirect_spec / ray_pdf)
                    bounce += 1
                    if bounce >= self.max_bounce:
                        break

                    if it.obj_id >= 0:                              # emission MIS for volpath
                        hit_light = self.emitter_id[it.obj_id]
                        if ti.static(self.use_mis):
                            emitter_pdf = 0.0
                            if hit_light >= 0 and self.is_delta(it.obj_id) == 0:
                                emitter_pdf = self.src_field[hit_light].solid_angle_pdf(it, ray_d)
                            emission_weight = balance_heuristic(ray_pdf, emitter_pdf)

                self.color[i, j] += ti.select(ti.math.isnan(color), 0., color)
                self.pixels[i, j] = self.color[i, j] / self.cnt[None]
            
    def summary(self):
        super().summary()
        CONSOLE.print(f"VPT SPP = {self.cnt[None]}. Rendering time: {self.clock.toc():.3f} s", justify="center")
        