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

from bxdf.medium import Medium
from scene.obj_desc import ObjDescriptor
from sampler.general_sampling import mis_weight

@ti.data_oriented
class VolumeRenderer(PathTracer):
    """
        Volumetric Renderer Final Class
    """
    def __init__(self, emitters: List[LightSource], objects: List[ObjDescriptor], prop: dict):
        super().__init__(emitters, objects, prop)
        # FIXME: calculate whether the world contains scattering medium
        self.world_scattering = False
        
    @ti.func
    def get_transmittance(self, idx: ti.i32, in_free_space: ti.i32, depth: ti.f32):
        valid_medium = False
        transmittance = vec3([1., 1., 1.])
        if in_free_space:
            valid_medium, transmittance = self.world.medium.transmittance(depth)
        else:
            # if not in_free_space, bsdf_field[idx] must be valid
            # TODO: check if any sample of BRDF can penetrate the surface without termination
            valid_medium, transmittance = self.bsdf_field[idx].medium.transmittance(depth)
        return valid_medium, transmittance

    @ti.func
    def sample_mfp(self, idx: ti.i32, in_free_space: ti.i32, depth: ti.f32):
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
            else:
                is_mi, mfp, beta = self.bsdf_field[idx].medium.sample_mfp(depth)
            # use medium to sample / calculate transmittance
        return is_mi, mfp, beta
    
    @ti.func
    def track_ray(self, ray, start_p, depth):
        """ 
            For medium interaction, check if the path to one point is not blocked (by non-null surface object)
            And also we need to calculate the attenuation along the path, e.g.: if the ray passes through
            two clouds of smoke and between the two clouds there is a transparent medium
            
            Here, we also keep tracks of what kind of medium we are in
        """
        tr = vec3([1., 1., 1.])
        for _i in range(5):             # maximum tracking depth = 5 (corresponding to at most 2 clouds of smoke)
            obj_id, normal, min_depth = self.ray_intersect(ray, start_p, depth)
            if obj_id < 0: break        # does not intersect anything - break (point source in the void with no medium)
            in_free_space = tm.dot(normal, ray) < 0
            valid_medium, transmittance = self.get_transmittance(obj_id, in_free_space, min_depth)
            # invalid medium can be "BRDF" or "transparent medium". Transparent medium has non-null surface, therefore invalid
            if not valid_medium:        # non-null surface blocks the ray path, break
                tr.fill(0.0)
                break  
            tr *= transmittance
            start_p += ray * min_depth
            depth -= min_depth
            if depth <= 5e-4: break     # reach the target point: break
        return tr
        
    @ti.kernel
    def render(self):
        self.cnt[None] += 1
        for i, j in self.pixels:
            # TODO: MIS in VPT is not considered yet (too complex)
            ray_d = self.pix2ray(i, j)
            ray_o = self.cam_t
            normal          = vec3([0, 1, 0])
            color           = vec3([0, 0, 0])
            throughput      = vec3([1, 1, 1])
            for _i in range(self.max_bounce):
                # Step 1: ray termination test - Only RR termination is allowed
                max_value = throughput.max()
                if ti.random(float) > max_value: break
                else: throughput *= 1. / ti.max(max_value, 1e-7)    # unbiased calculation
                # Step 2: ray intersection
                obj_id, normal, min_depth = self.ray_intersect(ray_d, ray_o)

                if obj_id < 0: break                                # nothing is hit, break
                ray_dot = tm.dot(normal, ray_d)
                # Step 3: check for mean free path sampling
                # Calculate mfp, path_beta = transmittance / PDF
                is_mi, min_depth, path_beta = self.sample_mfp(obj_id, ray_dot < 0, min_depth) 
                hit_point = ray_d * min_depth + ray_o
                hit_light = -1 if is_mi else self.emitter_id[obj_id]
                # Step 4: direct component estimation
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
                        emit_pos, shadow_int, direct_pdf = emitter.         \
                            sample(self.precom_vec, self.normals, self.mesh_cnt, hit_point)        # sample light
                        to_emitter  = emit_pos - hit_point
                        emitter_d   = to_emitter.norm()
                        light_dir   = to_emitter / emitter_d
                        tr = self.track_ray(ray_d, ray_o, emitter_d)
                        shadow_int *= tr
                        direct_spec = self.eval(obj_id, ray_d, light_dir, normal, self.world.medium, is_mi)
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
                ray_d, indirect_spec, ray_pdf = self.sample_new_ray(obj_id, ray_d, normal, self.world.medium, is_mi)
                ray_o = hit_point
                throughput *= path_beta         # attenuate first
                color += (direct_int + emit_int) * throughput
                throughput *= (indirect_spec / ray_pdf)

            self.color[i, j] += ti.select(ti.math.isnan(color), 0., color)
            self.pixels[i, j] = self.color[i, j] / self.cnt[None]
            
    def summary(self):
        print(f"[INFO] VPT Finished rendering. SPP = {self.cnt[None]}. Rendering time: {self.clock.toc():.3f} s")
        