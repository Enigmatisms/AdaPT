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
from scene.opts import get_options
from scene.obj_desc import ObjDescriptor
from scene.xml_parser import mitsuba_parsing
from renderer.constants import TRANSPORT_UNI, TRANSPORT_IMP

from sampler.general_sampling import *
from utils.tools import TicToc

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
        self.clock = TicToc()
        self.anti_alias         = prop['anti_alias']
        self.stratified_sample  = prop['stratified_sampling']   # whether to use stratified sampling
        self.use_mis            = prop['use_mis']               # whether to use multiple importance sampling
        self.num_shadow_ray     = prop['num_shadow_ray']        # number of shadow samples to trace
        if self.num_shadow_ray > 0:
            self.inv_num_shadow_ray = 1. / float(self.num_shadow_ray)
        else:
            self.inv_num_shadow_ray = 1.
        
        self.world              = prop['world'].export()        # world (free space / ambient light / background props)
        # for object with attached light source, emitter id stores the reference id to the emitter
        self.emitter_id = ti.field(int, self.num_objects)   
                     
        self.src_num    = len(emitters)
        self.color      = ti.Vector.field(3, float, (self.w, self.h))       # color without normalization
        self.src_field  = TaichiSource.field()
        self.brdf_field = BRDF.field()
        self.bsdf_field = BSDF.field()
        ti.root.dense(ti.i, self.src_num).place(self.src_field)             # Light source Taichi storage
        self.brdf_nodes = ti.root.bitmasked(ti.i, self.num_objects)
        self.brdf_nodes.place(self.brdf_field)                              # BRDF Taichi storage
        ti.root.bitmasked(ti.i, self.num_objects).place(self.bsdf_field)    # BRDF Taichi storage (no node needed)

        print(f"[INFO] Path tracer param loading in {self.clock.toc(True):.3f} ms")
        self.clock.tic()
        self.initialze(emitters, objects)
        print(f"[INFO] Path tracer initialization in {self.clock.toc(True):.3f} ms")
        self.clock.tic()

    def initialze(self, emitters: List[LightSource], objects: List[ObjDescriptor]):
        for i, emitter in enumerate(emitters):
            self.src_field[i] = emitter.export()
            self.src_field[i].obj_ref_id = -1
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
            if type(obj.bsdf) == BSDF_np:
                self.bsdf_field[i]  = obj.bsdf.export()
                print(f"Object [{i}] = {obj}, bxdf = {obj.bsdf}")
            else:
                self.brdf_field[i]  = obj.bsdf.export()
                print(f"Object [{i}] = {obj}, bxdf = {obj.bsdf}")
            self.aabbs[i, 0]    = vec3(obj.aabb[0])        # unrolled
            self.aabbs[i, 1]    = vec3(obj.aabb[1])
            emitter_ref_id      = obj.emitter_ref_id
            self.emitter_id[i]  = emitter_ref_id
            if emitter_ref_id  >= 0:
                self.src_field[emitter_ref_id].obj_ref_id = i

    @ti.func
    def sample_new_ray(self, idx: int, incid: vec3, normal: vec3, is_mi: int, in_free_space: int, mode: int = TRANSPORT_UNI):
        """ Mode is for cosine term calculation: \\
            For camera path, cosine term is computed against (ray_out and normal), \\
            while for light path, cosine term is computed against ray_in and normal \\
            This only affects surface interaction since medium interaction produce no cosine term
        """
        ret_dir  = vec3([0, 1, 0])
        ret_spec = vec3([1, 1, 1])
        ret_pdf  = 1.0
        if is_mi:
            if in_free_space:       # sample world medium
                ret_dir, ret_spec, ret_pdf = self.world.medium.sample_new_rays(incid)
            else:                   # sample object medium
                ret_dir, ret_spec, ret_pdf = self.bsdf_field[idx].medium.sample_new_rays(incid)
        else:                       # surface sampling
            if ti.is_active(self.brdf_nodes, idx):      # active means the object is attached to BRDF
                ret_dir, ret_spec, ret_pdf = self.brdf_field[idx].sample_new_rays(incid, normal)
            else:                                       # directly sample surface
                ret_dir, ret_spec, ret_pdf = self.bsdf_field[idx].sample_surf_rays(incid, normal, self.world.medium, mode)
        return ret_dir, ret_spec, ret_pdf

    @ti.func
    def eval(self, idx: int, incid: vec3, out: vec3, normal: vec3, is_mi: int, in_free_space: int, mode: int = TRANSPORT_UNI) -> vec3:
        ret_spec = vec3([1, 1, 1])
        if is_mi:
            # FIXME: eval_phase and phase function currently return a float
            if in_free_space:       # evaluate world medium
                ret_spec.fill(self.world.medium.eval(incid, out))
            else:                   # is_mi implys is_scattering = True
                ret_spec.fill(self.bsdf_field[idx].medium.eval(incid, out))
        else:                       # surface interaction
            if ti.is_active(self.brdf_nodes, idx):      # active means the object is attached to BRDF
                ret_spec = self.brdf_field[idx].eval(incid, out, normal)
            else:                                       # directly evaluate surface
                ret_spec = self.bsdf_field[idx].eval_surf(incid, out, normal, self.world.medium, mode)
        return ret_spec
    
    @ti.func
    def surface_pdf(self, idx: int, outdir: vec3, normal: vec3, incid: vec3):
        """ Outdir: actual incident ray direction, incid: ray (from camera) """
        pdf = 0.
        if ti.is_active(self.brdf_nodes, idx):      # active means the object is attached to BRDF
            pdf = self.brdf_field[idx].get_pdf(outdir, normal, incid)
        else:
            pdf = self.bsdf_field[idx].get_pdf(outdir, normal, incid, self.world.medium)
        return pdf
    
    @ti.func
    def get_pdf(self, idx: int, incid: vec3, out: vec3, normal: vec3, is_mi: int, in_free_space: int):
        pdf = 0.
        if is_mi:   # evaluate phase function
            if in_free_space:
                pdf = self.world.medium.eval(incid, out)
            else:
                pdf = self.bsdf_field[idx].medium.eval(incid, out)
        else:
            pdf = self.surface_pdf(idx, out, normal, incid)
        return pdf
    
    @ti.func
    def is_delta(self, idx: int):
        is_delta = False
        if idx >= 0:
            if ti.is_active(self.brdf_nodes, idx):      # active means the object is attached to BRDF
                is_delta = self.brdf_field[idx].is_delta
            else:
                is_delta = self.bsdf_field[idx].is_delta
        return is_delta
    
    @ti.func
    def get_bsdf_id(self, idx: int):
        ret_id = -1
        if idx >= 0:
            if ti.is_active(self.brdf_nodes, idx):      # active means the object is attached to BRDF
                ret_id = self.brdf_field[idx]._type
            else:
                ret_id = self.bsdf_field[idx]._type + 10
        return ret_id
    
    @ti.func
    def is_scattering(self, idx: int):           # check if the object with index idx is a scattering medium
        # FIXME: if sigma_t is too small, set the scattering medium to det-refract
        is_scattering = False
        if idx >= 0 and not ti.is_active(self.brdf_nodes, idx):
            is_scattering = self.bsdf_field[idx].medium.is_scattering()
        return is_scattering

    @ti.func
    def sample_light(self, no_sample: int = -1):
        """
            return selected light source, pdf and whether the current source is valid
            if can only sample <id = no_sample>, then the sampled source is invalid
            sample light might need to return more information (medium transmittance information)
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
        return self.src_field[idx], pdf, valid_sample, idx
    
    @ti.func
    def get_associated_obj(self, emit_id: int):
        return self.src_field[emit_id].obj_ref_id

if __name__ == "__main__":
    options = get_options()
    ti.init(arch = ti.vulkan, kernel_profiler = options.profile, default_ip = int, default_fp = float)
    emitter_configs, _, meshes, configs = mitsuba_parsing(options.input_path, options.scene)  # complex_cornell
    pt = PathTracer(emitter_configs, meshes, configs)