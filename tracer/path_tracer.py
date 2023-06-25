"""
    Path tracer for indirect / global illumination
    This module will be progressively built. Currently, participating media is not supported
    @author: Qianyue He
    @date: 2023.1.26
"""

import os
import sys
sys.path.append("..")

import tqdm
import numpy as np
import taichi as ti
import taichi.math as tm
from taichi.math import vec2, vec3

from typing import List
from la.cam_transform import *
from tracer.tracer_base import TracerBase
from emitters.abtract_source import LightSource, TaichiSource

from bxdf.brdf import BRDF
from bxdf.bsdf import BSDF, BSDF_np
from bxdf.texture import Texture, Texture_np
from parsers.opts import get_options
from parsers.obj_desc import ObjDescriptor
from parsers.xml_parser import scene_parsing
from renderer.constants import TRANSPORT_UNI, INV_2PI, INV_PI, INVALID

from sampler.general_sampling import *
from utils.tools import TicToc
from tracer.interaction import Interaction
from tracer.ti_bvh import LinearBVH, LinearNode, export_python_bvh

from rich.console import Console
CONSOLE = Console(width = 128)

"""
    2023-5-18: There is actually tremendous amount of work to do
    1. BRDF/BSDF should incorporate Texture, there is a way to by-pass this
        1. Pass the UV-queried color into the functions (could be painful)
        2. store the uv_color in the vertex (for evaluation during BDPT)
        This is by-far the simplest solution
    TODO: many APIs should be modified
"""

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

        # These four texture mappings might be accessed with the same pattern
        self.albedo_map    = Texture.field()
        self.normal_map    = Texture.field()
        self.bump_map      = Texture.field()

        # FIXME: I might need to dive deeper, this might not be appropriate
        # maybe we should opt for environment map (env lighting)
        self.roughness_map = Texture.field()                                # TODO: this is useless for now

        ti.root.dense(ti.i, self.src_num).place(self.src_field)             # Light source Taichi storage
        self.obj_nodes = ti.root.bitmasked(ti.i, self.num_objects)
        self.obj_nodes.place(self.brdf_field)                              # BRDF Taichi storage
        ti.root.bitmasked(ti.i, self.num_objects).place(self.bsdf_field)    # BRDF Taichi storage (no node needed)
        ti.root.dense(ti.i, self.num_objects).place(self.albedo_map, self.normal_map, self.bump_map, self.roughness_map)

        if prop["packed_textures"] is None:
            self.albedo_img    = ti.Vector.field(3, float, (1, 1))
            self.normal_img    = ti.Vector.field(3, float, (1, 1))
            self.bump_img      = ti.Vector.field(3, float, (1, 1))
            self.roughness_img = ti.Vector.field(3, float, (1, 1))
            self.has_albedo_map    = False
            self.has_normal_map    = False
            self.has_bump_map      = False
            self.has_roughness_map = False
        else:
            images = prop["packed_texture"]     # dict ('albedo': Optional(), 'normal': ...)
            for key, image in images.item():
                if image is None:               # no image means we don't have this kind of mapping
                    self.__setattr__(f"{key}_img", ti.Vector.field(3, float, (1, 1)))
                    continue
                tex_h, tex_w, _  = image.shape
                self.__setattr__(f"has_{key}_map", True)
                self.__setattr__(f"{key}_img", ti.Vector.field(3, float, (tex_w, tex_h)))
                self.__getattribute__(f"{key}_img").from_numpy(image)
                CONSOLE.log(f"Packed texture image tagged '{key}' loaded: ({tex_w}, {tex_h})")
        
        CONSOLE.log(f"Path tracer param loading in {self.clock.toc_tic(True):.3f} ms")
        self.initialze(emitters, objects)
        CONSOLE.log(f"Path tracer initialization in {self.clock.toc(True):.3f} ms")

        min_val = vec3([1e3, 1e3, 1e3])
        max_val = vec3([-1e3, -1e3, -1e3])
        for i in range(self.num_objects):
            min_val = ti.min(min_val, self.aabbs[i, 0])
            max_val = ti.max(max_val, self.aabbs[i, 1])
        
        # world aabb for unbounded scene
        self.w_aabb_min = ti.min(self.cam_t, min_val) - 0.1         
        self.w_aabb_max = ti.max(self.cam_t, max_val) + 0.1

        if prop.get('accelerator', 'none') == 'bvh':
            try:
                from bvh_cpp import bvh_build
            except ImportError:
                CONSOLE.log(":warning: Warning: [bold green]pybind11 BVH cpp[/bold green] is not built. Please check whether you have compiled bvh_cpp module.")
                CONSOLE.log("[yellow]:warning: Warning: Fall back to brute force primitive traversal.")
            else:
                CONSOLE.log(":rocket: Using SAH-BVH tree accelerator.")
                primitives, obj_info = self.prepare_for_bvh(objects)
                self.clock.tic()
                py_nodes, py_bvhs = bvh_build(primitives, obj_info, self.w_aabb_min.to_numpy(), self.w_aabb_max.to_numpy())
                CONSOLE.log(f":rocket: BVH construction finished in {self.clock.toc_tic(True):.3f} ms")

                self.node_num = len(py_nodes)
                self.bvh_num = len(py_bvhs)

                self.lin_nodes = LinearNode.field()
                self.lin_bvhs  = LinearBVH.field()
                ti.root.dense(ti.i, self.node_num).place(self.lin_nodes)
                ti.root.dense(ti.i, self.bvh_num).place(self.lin_bvhs)
                export_python_bvh(self.lin_nodes, self.lin_bvhs, py_nodes, py_bvhs)
                CONSOLE.log(f"{self.node_num } nodes and {self.bvh_num} bvh primitives are loaded.")
                self.__setattr__("ray_intersect", self.ray_intersect_bvh)
                self.__setattr__("does_intersect", self.does_intersect_bvh)

    def get_check_point(self):
        """This is a simple checkpoint saver, which can be hacked.
           I do not offer strict consistency checking since this is laborious
        """
        items_to_save = ["w", "h", "crop_x", "crop_y", "crop_rx", "crop_ry", "focal"]
        items_to_save += ["num_objects", "num_prims", "cam_orient", "src_num"]
        check_point = {}
        for item in items_to_save:
            check_point[item] = getattr(self, item)
        check_point["cam_t"] = self.cam_t.to_numpy()
        check_point["accumulation"] = self.color.to_numpy()
        check_point["counter"] = self.cnt[None]
        return check_point
    
    def load_check_point(self, check_point: dict):
        """ Compare some basic configs (for consistency), if passed
            load the information into the current renderer
        """
        for key, val in check_point.items():
            if key not in {"accumulation", "counter", "cam_t", "cam_orient"}:
                if val == getattr(self, key): continue
            elif key == "cam_t":
                if np.abs(val - self.cam_t.to_numpy()).max() < 1e-4: continue
            elif key == "cam_orient":
                if np.abs(val - self.cam_orient).max() < 1e-4: continue
            else: continue
            CONSOLE.log(f"[bold red]:skull: Error: '{key}' from the checkpoint is different.")
            exit(1)
        CONSOLE.log(f"[bold green]Recovered from check-point, elapsed counter: {check_point['counter']}")
        self.color.from_numpy(check_point["accumulation"])
        self.cnt[None] = check_point["counter"]

    def prepare_for_bvh(self, objects: List[ObjDescriptor]):
        primitives = []
        obj_info = np.zeros((2, len(objects)), dtype = np.int32)        
        for i, obj in tqdm.tqdm(enumerate(objects)):
            # for sphere, it would be (1, 2, 3), for others it would be (n, 3, 3)
            num_primitive = obj.meshes.shape[0]
            obj_info[0, i] = num_primitive
            obj_info[1, i] = obj.type
            for primitive in obj.meshes:
                if primitive.shape[0] < 3:
                    primitive = np.vstack((primitive, np.zeros((1, 3), dtype=np.float32)))
                primitives.append(primitive)
        primitives = np.stack(primitives, axis = 0).astype(np.float32)
        return primitives, obj_info

    def initialze(self, emitters: List[LightSource], objects: List[ObjDescriptor]):
        # FIXME: Path tracer initialization is too slow
        self.uv_coords.fill(0)
        for i, emitter in enumerate(emitters):
            self.src_field[i] = emitter.export()
            self.src_field[i].obj_ref_id = -1

        acc_prim_num = 0
        for i, obj in enumerate(objects):
            prim_num = obj.meshes.shape[0]
            for j, (mesh, normal) in tqdm.tqdm(enumerate(zip(obj.meshes, obj.normals)), total = prim_num):
                cur_id = acc_prim_num + j
                self.prims[cur_id, 0] = vec3(mesh[0])
                self.prims[cur_id, 1] = vec3(mesh[1])
                if mesh.shape[0] > 2:       # not a sphere
                    self.prims[cur_id, 2] = vec3(mesh[2])
                    self.precom_vec[cur_id, 0] = self.prims[cur_id, 1] - self.prims[cur_id, 0]                    
                    self.precom_vec[cur_id, 1] = self.prims[cur_id, 2] - self.prims[cur_id, 0] 
                    self.precom_vec[cur_id, 2] = self.prims[cur_id, 0]
                else:
                    self.precom_vec[cur_id, 0] = self.prims[cur_id, 0]
                    self.precom_vec[cur_id, 1] = self.prims[cur_id, 1]           
                if obj.vns is not None:             # got vertex normal
                    pass
                self.normals[cur_id] = vec3(normal) 
                if obj.uv_coords is not None:
                    uv_coord = obj.uv_coords[j]
                    self.uv_coords[cur_id, 0] = vec2(uv_coord[0])
                    self.uv_coords[cur_id, 1] = vec2(uv_coord[1])
                    self.uv_coords[cur_id, 2] = vec2(uv_coord[2])
                if obj.vns is not None:
                    vn = obj.vns[j]
                    self.v_normals[cur_id, 0] = vec3(vn[0])
                    self.v_normals[cur_id, 1] = vec3(vn[1])
                    self.v_normals[cur_id, 2] = vec3(vn[2])

            self.obj_info[i, 0] = acc_prim_num
            self.obj_info[i, 1] = obj.tri_num
            self.obj_info[i, 2] = obj.type
            acc_prim_num        += obj.tri_num

            if type(obj.bsdf) == BSDF_np:
                self.bsdf_field[i]  = obj.bsdf.export()
            else:
                self.brdf_field[i]  = obj.bsdf.export()
            
            for key, value in obj.texture_group.items():        # exporting texture group
                if value is not None:
                    self.__getattribute__(f"{key}_map")[i] = value.export()
                else:
                    self.__getattribute__(f"{key}_map")[i] = Texture_np.default()
                    
            self.aabbs[i, 0]    = vec3(obj.aabb[0])        # unrolled
            self.aabbs[i, 1]    = vec3(obj.aabb[1])
            emitter_ref_id      = obj.emitter_ref_id
            self.emitter_id[i]  = emitter_ref_id
            if emitter_ref_id  >= 0:
                self.src_field[emitter_ref_id].obj_ref_id = i

    @ti.func
    def get_uv_item(self, textures: ti.template(), tex_img: ti.template(), it: ti.template()):
        """ Convert primitive local UV to the global UV coord for an object """
        tex_value = INVALID
        valid = False
        if it.obj_id >= 0 and textures[it.obj_id].type > -255:
            u, v = it.uv                # local u and local v
            is_sphere = self.obj_info[it.obj_id, 2]
            if is_sphere == 0:          # not a sphere
                u, v = self.uv_coords[it.prim_id, 1] * u + self.uv_coords[it.prim_id, 2] * v + \
                    self.uv_coords[it.prim_id, 0] * (1. - u - v)        # convert from local to global
            tex_value = textures[it.obj_id].query(tex_img, u, v)
            valid = True
        return tex_value, valid

    @ti.func
    def process_ns(self, it: ti.template()):
        """ Process shading normal for interaction 
            For example, we may have bump map / normal map
            We should convert the normal from local frame to global frame
            normal map: replace the original shading normal with normal map normal
            bump map: apply a rotational offset to the current shading normal
        """
        if ti.static(self.has_normal_map):
            normal, n_valid = self.get_uv_item(self.normal_map, self.normal_img, it)
            if n_valid:
                R = rotation_between(vec3([0, 1, 0]), it.n_g)
                it.n_s = (R @ normal).normalized()
        if ti.static(self.has_bump_map):
            delta_n, d_valid = self.get_uv_item(self.bump_map, self.bump_img, it)
            if d_valid:
                it.n_s, _ = delocalize_rotate(it.n_s, delta_n)

    @ti.func
    def bvh_intersect(self, bvh_id, ray, start_p):
        """ Intersect Ray with a BVH node """
        obj_idx, prim_idx = self.lin_bvhs[bvh_id].get_info()
        is_sphere = self.obj_info[obj_idx, 2]
        ray_t = -1.
        u = 0.
        v = 0.
        if is_sphere > 0:
            center  = self.prims[prim_idx, 0]
            radius2 = self.prims[prim_idx, 1][0] ** 2
            s2c     = center - start_p
            center_norm2 = s2c.norm_sqr()
            proj_norm = tm.dot(ray, s2c)
            c2ray_norm = center_norm2 - proj_norm ** 2  # center to ray distance ** 2
            if c2ray_norm < radius2:
                ray_t = proj_norm
                ray_cut = ti.sqrt(radius2 - c2ray_norm)
                ray_t += ti.select(center_norm2 > radius2 + 1e-4, -ray_cut, ray_cut)
        else:
            p1 = self.prims[prim_idx, 0]
            v1 = self.precom_vec[prim_idx, 0]
            v2 = self.precom_vec[prim_idx, 1]
            mat = ti.Matrix.cols([v1, v2, -ray]).inverse()
            u, v, t = mat @ (start_p - p1)
            # u, v as barycentric coordinates should be returned
            if u >= 0 and v >= 0 and u + v <= 1.0:
                ray_t = t
        return ray_t, obj_idx, prim_idx, is_sphere, u, v
    
    @ti.func
    def ray_intersect_bvh(self, ray: vec3, start_p, min_depth = -1.0):
        """ Ray intersection with BVH, to prune unnecessary computation """
        obj_id  = -1
        prim_id = -1
        sphere_flag = False
        min_depth = ti.select(min_depth > 0.0, min_depth - 1e-4, 1e7)
        node_idx = 0
        inv_ray = 1. / ray
        coord_u = 0.
        coord_v = 0.
        while node_idx < self.node_num:
            aabb_intersect, t_near = self.lin_nodes[node_idx].aabb_test(inv_ray, start_p)
            if aabb_intersect == False or t_near > min_depth: 
                # if the current node is not intersected, then all of the following nodes can be skipped
                node_idx += self.lin_nodes[node_idx].all_offset         # skip the entire node (and sub-tree)
                continue
            # otherwise, the current node should be investigated, check if it's leaf node
            if self.lin_nodes[node_idx].is_leaf():
                # Traverse all the primtives in the leaf, update min_depth, obj_id and tri_id
                begin_i, end_i = self.lin_nodes[node_idx].get_range() 
                for bvh_i in range(begin_i, end_i):
                    aabb_intersect, t_near = self.lin_bvhs[bvh_i].aabb_test(inv_ray, start_p)
                    if aabb_intersect == False or t_near > min_depth: 
                        continue
                    ray_t, obj_idx, prim_idx, obj_type, u, v = self.bvh_intersect(bvh_i, ray, start_p)
                    if ray_t > 1e-4 and ray_t < min_depth:
                        min_depth   = ray_t
                        obj_id      = obj_idx
                        prim_id     = prim_idx
                        sphere_flag = obj_type
                        coord_u     = u
                        coord_v     = v
            node_idx += 1
        n_g = vec3([1, 0, 0])
        n_s = vec3([1, 0, 0])
        if obj_id >= 0:
            if sphere_flag:
                center = self.prims[prim_id, 0]
                n_g = (start_p + min_depth * ray - center).normalized() 
                coord_u = (tm.atan2(n_g[1], n_g[0]) + tm.pi) * INV_2PI
                coord_v = tm.acos(n_g[2]) * INV_PI
                n_s = n_g
            else:
                n_g = self.normals[prim_id]
                # calculate vertex normal (for shading) if vertex normal exists
                if ti.static(self.has_v_normal):
                    n_s = self.v_normals[prim_id, 0] * (1. - coord_u - coord_v) + \
                        coord_u * self.v_normals[prim_id, 1] + \
                        coord_v * self.v_normals[prim_id, 2]
                else: 
                    n_s = n_g
        # The returned coord_u and coord_v is not the actual uv coords (but for one specific primitive)
        return Interaction(
            obj_id = obj_id, prim_id = prim_id, n_g = n_g, n_s = n_s,
            uv = vec2(coord_u, coord_v), min_depth = min_depth
        )
    
    @ti.func
    def does_intersect_bvh(self, ray, start_p, min_depth = -1.0):
        """ Ray intersection with BVH, to prune unnecessary computation """
        node_idx = 0
        hit_flag = False
        min_depth = ti.select(min_depth > 0.0, min_depth - 1e-4, 1e7)
        inv_ray = 1. / ray
        while node_idx < self.node_num:
            aabb_intersect, t_near = self.lin_nodes[node_idx].aabb_test(inv_ray, start_p)
            if aabb_intersect == False or t_near > min_depth: 
                # if the current node is not intersected, then all of the following nodes can be skipped
                node_idx += self.lin_nodes[node_idx].all_offset         # skip the entire node (and sub-tree)
                continue
            # otherwise, the current node should be investigated, check if it's leaf node
            if self.lin_nodes[node_idx].is_leaf():
                # Traverse all the primtives in the leaf, update min_depth, obj_id and tri_id
                begin_i, end_i = self.lin_nodes[node_idx].get_range() 
                for bvh_i in range(begin_i, end_i):
                    aabb_intersect, t_near = self.lin_bvhs[bvh_i].aabb_test(inv_ray, start_p)
                    if aabb_intersect == False or t_near > min_depth: continue
                    ray_t, _i, _p, _o, _u, _v = self.bvh_intersect(bvh_i, ray, start_p)
                    if ray_t > 1e-4 and ray_t < min_depth:
                        hit_flag = True
                        break
            if hit_flag: break
            node_idx += 1
        return hit_flag

    @ti.func
    def sample_new_ray(self, 
        it: ti.template(), incid: vec3, is_mi: int, 
        in_free_space: int, mode: int = TRANSPORT_UNI
    ):
        """ Mode is for cosine term calculation: \\
            For camera path, cosine term is computed against (ray_out and normal), \\
            while for light path, cosine term is computed against ray_in and normal \\
            This only affects surface interaction since medium interaction produce no cosine term
            Note 2023.5.21: Actually, passing color vec3 from the outside makes me feel sick...
        """
        ret_dir  = vec3([0, 1, 0])
        ret_spec = vec3([1, 1, 1])
        ret_pdf  = 1.0
        if is_mi:
            if in_free_space:       # sample world medium
                ret_dir, ret_spec, ret_pdf = self.world.medium.sample_new_rays(incid)
            else:                   # sample object medium
                ret_dir, ret_spec, ret_pdf = self.bsdf_field[it.obj_id].medium.sample_new_rays(incid)
        else:                       # surface sampling
            if ti.is_active(self.obj_nodes, it.obj_id):      # active means the object is attached to BRDF
                ret_dir, ret_spec, ret_pdf = self.brdf_field[it.obj_id].sample_new_rays(it, incid)
            else:                                       # directly sample surface
                ret_dir, ret_spec, ret_pdf = self.bsdf_field[it.obj_id].sample_surf_rays(it, incid, self.world.medium, mode)
        return ret_dir, ret_spec, ret_pdf

    @ti.func
    def eval(self, it: ti.template(), incid: vec3, out: vec3, is_mi: int, 
             in_free_space: int, mode: int = TRANSPORT_UNI) -> vec3:
        ret_spec = vec3([1, 1, 1])
        if is_mi:
            # FIXME: eval_phase and phase function currently return a float
            if in_free_space:       # evaluate world medium
                ret_spec.fill(self.world.medium.eval(incid, out))
            else:                   # is_mi implys is_scattering = True
                ret_spec.fill(self.bsdf_field[it.obj_id].medium.eval(incid, out))
        else:                       # surface interaction
            if ti.is_active(self.obj_nodes, it.obj_id):      # active means the object is attached to BRDF
                ret_spec = self.brdf_field[it.obj_id].eval(it, incid, out)
            else:                                       # directly evaluate surface
                ret_spec = self.bsdf_field[it.obj_id].eval_surf(it, incid, out, self.world.medium, mode)
        return ret_spec
    
    @ti.func
    def surface_pdf(self, it: ti.template(), outdir: vec3, incid: vec3):
        """ Outdir: actual incident ray direction, incid: ray (from camera) """
        pdf = 0.
        if ti.is_active(self.obj_nodes, it.obj_id):      # active means the object is attached to BRDF
            pdf = self.brdf_field[it.obj_id].get_pdf(it, outdir, incid)
        else:
            pdf = self.bsdf_field[it.obj_id].get_pdf(it, outdir, incid, self.world.medium)
        return pdf
    
    @ti.func
    def get_pdf(self, it: ti.template(), incid: vec3, out: vec3, is_mi: int, in_free_space: int):
        pdf = 0.
        if is_mi:   # evaluate phase function
            if in_free_space:
                pdf = self.world.medium.eval(incid, out)
            else:
                pdf = self.bsdf_field[it.obj_id].medium.eval(incid, out)
        else:
            pdf = self.surface_pdf(it, it.obj_id, out, incid)
        return pdf
    
    @ti.func
    def get_ior(self, idx: int, in_free_space: int):
        # REAL FUNC
        ior = 1.
        if in_free_space: ior = self.world.medium.ior
        else: 
            if idx >= 0: ior = self.bsdf_field[idx].medium.ior
        return ior
    
    @ti.func
    def is_delta(self, idx: int):
        # REAL FUNC
        is_delta = False
        if idx >= 0:
            if ti.is_active(self.obj_nodes, idx):      # active means the object is attached to BRDF
                is_delta = self.brdf_field[idx].is_delta
            else:
                is_delta = self.bsdf_field[idx].is_delta
        return is_delta
    
    @ti.func
    def is_scattering(self, idx: int):           # check if the object with index idx is a scattering medium
        # FIXME: if sigma_t is too small, set the scattering medium to det-refract
        # REAL FUNC
        is_scattering = False
        if idx >= 0 and not ti.is_active(self.obj_nodes, idx):
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
    
    def summary(self):
        CONSOLE.rule()
        CONSOLE.print("[bold blue]:tada: :tada: :tada: Rendering Finished :tada: :tada: :tada:", justify="center")

if __name__ == "__main__":
    options = get_options()
    ti.init(arch = ti.vulkan, kernel_profiler = options.profile, default_ip = ti.int32, default_fp = ti.f32)
    input_folder = os.path.join(options.input_path, options.scene)
    emitter_configs, meshes, configs = scene_parsing(input_folder, options.name)  # complex_cornell
    pt = PathTracer(emitter_configs, meshes, configs)