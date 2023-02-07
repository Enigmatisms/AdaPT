"""
    Path tracer for indirect / global illumination
    This module will be progressively built. Currently, participating media is not supported
    @author: Qianyue He
    @date: 2023.1.26
"""

import sys
sys.path.append("..")

import taichi as ti
from taichi.math import vec3

from typing import List
from la.cam_transform import *
from tracer.tracer_base import TracerBase
from emitters.abtract_source import LightSource, TaichiSource

from bxdf.brdf import BRDF
from bxdf.bsdf import BSDF, BSDF_np
from scene.world import World
from scene.opts import get_options
from scene.obj_desc import ObjDescriptor
from scene.xml_parser import mitsuba_parsing

from sampler.general_sampling import *

@ti.data_oriented
class PathTracer(TracerBase):
    """
        Simple Ray tracing using Bary-centric coordinates
        This tracer can yield result with global illumination effect
    """
    def __init__(self, emitters: List[LightSource], objects: List[ObjDescriptor], prop: dict):
        super().__init__(objects, prop)
        """
            Implement path tracing algorithms first, then we can improve light source / BSDF / participating media
        """
        self.anti_alias         = prop['anti_alias']
        self.stratified_sample  = prop['stratified_sampling']   # whether to use stratified sampling
        self.use_mis            = prop['use_mis']               # whether to use multiple importance sampling
        self.num_shadow_ray     = prop['num_shadow_ray']        # number of shadow samples to trace
        assert(self.num_shadow_ray >= 1)
        self.inv_num_shadow_ray = 1. / float(self.num_shadow_ray)
        
        self.world              = prop['world'].export()        # world (free space / ambient light / background props)
        # for object with attached light source, emitter id stores the reference id to the emitter
        self.emitter_id = ti.field(ti.i32, self.num_objects)   
                     
        self.emit_max   = 1.0
        self.src_num    = len(emitters)
        self.color      = ti.Vector.field(3, ti.f32, (self.w, self.h))      # color without normalization
        self.src_field  = TaichiSource.field()
        self.brdf_field = BRDF.field()
        self.bsdf_field = BSDF.field()
        ti.root.dense(ti.i, self.src_num).place(self.src_field)             # Light source Taichi storage
        self.brdf_nodes = ti.root.bitmasked(ti.i, self.num_objects)
        self.brdf_nodes.place(self.brdf_field)                              # BRDF Taichi storage
        ti.root.bitmasked(ti.i, self.num_objects).place(self.bsdf_field)    # BRDF Taichi storage (no node needed)

        self.initialze(emitters, objects)

    def initialze(self, emitters: List[LightSource], objects: List[ObjDescriptor]):
        for i, emitter in enumerate(emitters):
            self.src_field[i] = emitter.export()
            self.src_field[i].obj_ref_id = -1
            self.emit_max = max(emitter.intensity.max(), self.emit_max)
        for i, obj in enumerate(objects):
            for j, (mesh, normal) in enumerate(zip(obj.meshes, obj.normals)):
                self.normals[i, j] = ti.Vector(normal) 
                for k, vec in enumerate(mesh):
                    self.meshes[i, j, k]  = ti.Vector(vec)
                if mesh.shape[0] > 2:       # not a sphere
                    self.precom_vec[i, j, 0] = self.meshes[i, j, 1] - self.meshes[i, j, 0]                    
                    self.precom_vec[i, j, 1] = self.meshes[i, j, 2] - self.meshes[i, j, 0]             
                    self.precom_vec[i, j, 2] = self.meshes[i, j, 0]
                else:
                    self.precom_vec[i, j, 0] = self.meshes[i, j, 0]
                    self.precom_vec[i, j, 1] = self.meshes[i, j, 1]
            self.mesh_cnt[i]    = obj.tri_num
            if type(obj.bsdf) == BSDF_np:
                self.bsdf_field[i]  = obj.bsdf.export()
            else:
                self.brdf_field[i]  = obj.bsdf.export()
            self.aabbs[i, 0]    = ti.Matrix(obj.aabb[0])        # unrolled
            self.aabbs[i, 1]    = ti.Matrix(obj.aabb[1])
            emitter_ref_id      = obj.emitter_ref_id
            self.emitter_id[i]  = emitter_ref_id
            if emitter_ref_id  >= 0:
                self.src_field[emitter_ref_id].obj_ref_id = i

    @ti.func
    def sample_new_ray(self, idx: ti.i32, incid: vec3, normal: vec3, medium):
        ret_dir  = vec3([0, 1, 0])
        ret_spec = vec3([1, 1, 1])
        pdf      = 1.0
        if ti.is_active(self.brdf_nodes, idx):      # active means the object is attached to BRDF
            ret_dir, ret_spec, pdf = self.brdf_field[idx].sample_new_rays(incid, normal, medium)
        else:
            ret_dir, ret_spec, pdf = self.bsdf_field[idx].sample_new_rays(incid, normal, medium)
        return ret_dir, ret_spec, pdf
    
    @ti.func
    def eval(self, idx: ti.i32, incid: vec3, out: vec3, normal: vec3, medium) -> vec3:
        ret_spec = vec3([1, 1, 1])
        if ti.is_active(self.brdf_nodes, idx):      # active means the object is attached to BRDF
            ret_spec = self.brdf_field[idx].eval(incid, out, normal, medium)
        else:
            ret_spec = self.bsdf_field[idx].eval(incid, out, normal, medium)
        return ret_spec
    
    @ti.func
    def get_pdf(self, idx: ti.i32, outdir: vec3, normal: vec3, incid: vec3, medium):
        pdf = 0.
        if ti.is_active(self.brdf_nodes, idx):      # active means the object is attached to BRDF
            pdf = self.brdf_field[idx].get_pdf(outdir, normal, incid, medium)
        else:
            pdf = self.bsdf_field[idx].get_pdf(outdir, normal, incid, medium)
        return pdf
    
    @ti.func
    def is_delta(self, idx: ti.i32):
        is_delta = False
        if ti.is_active(self.brdf_nodes, idx):      # active means the object is attached to BRDF
            is_delta = self.brdf_field[idx].is_delta
        else:
            is_delta = self.bsdf_field[idx].is_delta
        return is_delta

    @ti.func
    def sample_light(self, no_sample: ti.i32):
        """
            return selected light source, pdf and whether the current source is valid
            if can only sample <id = no_sample>, then the sampled source is invalid
        """
        idx = ti.random(int) % self.src_num
        pdf = 1. / self.src_num
        valid_sample = True
        if no_sample >= 0:
            if ti.static(self.src_num <= 1):
                valid_sample = False
            else:
                idx = ti.random(int) % (self.src_num - 1)
                if idx >= no_sample: idx += 1
                pdf = 1. / float(self.src_num - 1)
        return self.src_field[idx], pdf, valid_sample

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
                        emit_pos, shadow_int, direct_pdf = emitter.         \
                            sample(self.precom_vec, self.normals, self.mesh_cnt, hit_point)        # sample light
                        to_emitter  = emit_pos - hit_point
                        emitter_d   = to_emitter.norm()
                        light_dir   = to_emitter / emitter_d
                        if self.does_intersect(light_dir, hit_point, emitter_d):        # shadow ray 
                            shadow_int.fill(0.0)
                        else:
                            direct_spec = self.eval(obj_id, ray_d, light_dir, normal, self.world.medium)
                    else:       # the only situation for being invalid, is when there is only one source and the ray hit the source
                        break_flag = True
                        break
                    light_pdf = emitter_pdf * direct_pdf
                    if ti.static(self.use_mis):
                        bsdf_pdf = self.get_pdf(obj_id, light_dir, normal, ray_d, self.world.medium)
                        mis_w    = mis_weight(light_pdf, bsdf_pdf)
                        direct_int += direct_spec * shadow_int * mis_w
                    else:
                        direct_int += direct_spec * shadow_int / light_pdf
                if not break_flag:
                    direct_int *= self.inv_num_shadow_ray
                # emission: ray hitting an area light source
                emit_int    = vec3([0, 0, 0])
                if hit_light >= 0:
                    emit_int = self.src_field[hit_light].eval_le(hit_point - ray_o, normal)
                
                # indirect component requires sampling 
                ray_d, indirect_spec, ray_pdf = self.sample_new_ray(obj_id, ray_d, normal, self.world.medium)
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
                        emission_weight = mis_weight(ray_pdf, emitter_pdf)

            self.color[i, j] += ti.select(ti.math.isnan(color), 0., color)
            self.pixels[i, j] = self.color[i, j] / self.cnt[None]

if __name__ == "__main__":
    options = get_options()
    ti.init(arch = ti.vulkan, kernel_profiler = options.profile, default_ip = ti.i32, default_fp = ti.f32)
    emitter_configs, _, meshes, configs = mitsuba_parsing(options.input_path, options.scene)  # complex_cornell
    pt = PathTracer(emitter_configs, meshes, configs)
    gui = ti.GUI('Path Tracing', (pt.w, pt.h))
    iter_cnt = 0
    while True:
        for e in gui.get_events(gui.PRESS):
            if e.key == gui.ESCAPE:
                gui.running = False
        pt.render()
        gui.set_image(pt.pixels)
        gui.show()
        iter_cnt += 1
        if options.iter_num > 0 and iter_cnt > options.iter_num: break
        if gui.running == False: break
        gui.clear()
        pt.reset()

    if options.profile:
        ti.profiler.print_kernel_profiler_info() 
    pixels = pt.pixels.to_numpy()
    print(f"Maximum value: {pixels.max():.5f}")
    ti.tools.imwrite(pixels, options.output_path + options.img_name)
