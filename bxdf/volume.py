"""
    Grid Volume Scattering Medium (RGB / Float)
    @author: Qianyue He
    @date: 2024-6-22
"""

import os
import numpy as np
import taichi as ti
import xml.etree.ElementTree as xet

from typing import Tuple
from taichi.math import vec2, vec3, vec4, mat3
from bxdf.phase import PhaseFunction
from bxdf.medium import Medium_np
from la.cam_transform import delocalize_rotate
from parsers.general_parser import get, rgb_parse, transform_parse
from sampler.general_sampling import random_rgb
from renderer.constants import ZERO_V3, ONES_V3, EPSL_V3
from rich.console import Console
CONSOLE = Console(width = 128)

vec3i = ti.types.vector(3, int)

try:
    from vol_loader import vol_file_to_numpy
except ImportError:
    CONSOLE.log("[red]:skeleton: Error [/red]: vol_loader is not found, possible reason:")
    CONSOLE.log(f"module not installed. Use 'python ./setup.py install --user' in {'./bxdf/'}")
    raise ImportError("vol_loader not found")

__all__ = ["GridVolume_np", "GridVolume"]

FORCE_MONO_COLOR = False

""" TODO: add transform parsing
"""
class GridVolume_np:
    NONE = 0
    MONO = 1
    RGB  = 2
    __type_mapping = {"none": NONE, "mono": MONO, "rgb": RGB}
    def __init__(self, elem: xet.Element):
        self.albedo = np.ones(3, np.float32)
        self.par = np.zeros(3, np.float32)
        self.pdf = np.float32([1., 0., 0.])
        self.phase_type_id = -1
        self.phase_type = "hg"
        self.type_name = "none"

        self.type_id = GridVolume_np.NONE
        self.xres = 0
        self.yres = 0
        self.zres = 0
        self.channel = 0

        self.density_grid = None

        self.toWorld = None
        self.scale = None
        self.rotation = np.eye(3, dtype = np.float32)
        self.offset = np.zeros(3, dtype = np.float32)
        self.forward_t = None
        self.density_scaling = np.ones(3, dtype = np.float32)

        # phase function
        self.par = np.zeros(3, np.float32)
        self.pdf = np.float32([1., 0., 0.])

        elem_to_query = {
            "rgb": rgb_parse, 
            "float": lambda el: get(el, "value"), 
            "string": lambda el: self.setup_volume(get(el, "path", str)),
            "transform": lambda el: self.assign_transform(*transform_parse(el))
        }
        if elem is not None:
            type_name = elem.get("type")
            if type_name in GridVolume_np.__type_mapping:
                self.type_id = GridVolume_np.__type_mapping[type_name]
            else:
                CONSOLE.log(f"[red]Error :skull:[/red]: Volume '{elem.get('name')}' has unsupported type '{type_name}'")
                raise NotImplementedError(f"GridVolume type '{type_name}' is not supported.")
            self.type_name = type_name 

            phase_type = elem.get("phase_type")
            _phase_type_id = Medium_np.is_supported_type(phase_type)
            if _phase_type_id is not None:
                self.phase_type_id = _phase_type_id
            else:
                raise NotImplementedError(f"Phase function type '{phase_type}' is not supported.")
            
            for tag, query_func in elem_to_query.items():
                tag_elems = elem.findall(tag)
                for tag_elem in tag_elems:
                    name = tag_elem.get("name")
                    if hasattr(self, name):
                        self.__setattr__(name, query_func(tag_elem))
            
            if self.scale is None:
                self.scale = np.eye(3, dtype = np.float32)
            else:
                self.scale = np.diag(self.scale)
            self.forward_t = self.rotation @ self.scale
            if self.type_id == GridVolume_np.MONO:
                self.density_grid *= self.density_scaling[0]
            else:
                print(f"Shape: {self.density_grid.shape}, {self.density_scaling}")
                self.density_grid *= self.density_scaling

    def assign_transform(self, trans_r, trans_t, trans_s):
        self.rotation = trans_r
        self.offset   = trans_t
        self.scale    = trans_s

    def setup_volume(self, path:str):
        if not os.path.exists(path):
            CONSOLE.log(f"[red]Error :skull:[/red]: {path} contains no valid volume file.")
            raise RuntimeError("Volume file not found.")
        density_grid, (self.xres, self.yres, self.zres, self.channel) = vol_file_to_numpy(path, FORCE_MONO_COLOR)
        density_grid = GridVolume_np.make_rgb_volume_from_monocolor(density_grid)
        self.channel = 3
        if FORCE_MONO_COLOR:
            CONSOLE.log(f"[yellow]Warning[/yellow]: FORCE_MONO_COLOR is True. This only makes sense when we are testing the code.")
        return density_grid.reshape((self.zres, self.yres, self.xres, self.channel))
    
    @staticmethod
    def make_rgb_volume_from_monocolor(density_grid):
        density_grid = np.stack([density_grid, density_grid * 0.2, density_grid * 0.2], axis = -1)
        return density_grid
    
    def get_shape(self) -> Tuple[int, int, int]:
        return (self.zres, self.yres, self.xres)

    def export(self):
        if self.type_id == GridVolume_np.NONE:
            return GridVolume(_type = 0)
        aabb_mini, aabb_maxi = self.get_aabb()
        maj = self.density_grid.max(axis = (0, 1, 2))         # the shape of density grid: (zres, yres, xres, channels)
        return GridVolume(
            _type   = self.type_id,
            albedo  = vec3(self.albedo),
            inv_T   = mat3(np.linalg.inv(self.forward_t)),
            trans   = vec3(self.offset),
            mini    = vec3(aabb_mini), 
            maxi    = vec3(aabb_maxi),
            max_idxs = vec3i([self.xres, self.yres, self.zres]),
            majorant = vec3(maj) if self.type_id == GridVolume_np.RGB else vec3([maj, maj, maj]),
            ph       = PhaseFunction(_type = self.phase_type_id, par = vec3(self.par), pdf = vec3(self.pdf))
        )
    
    def local_to_world(self, point: np.ndarray) -> np.ndarray:
        """ Take a point (shape (3) and transform it to the world space) """
        return point @ self.forward_t.T + self.offset 
    
    def get_aabb(self):
        x, y, z = self.xres, self.yres, self.zres
        all_points = np.float32([
            [0, 0, 0],
            [x, 0, 0],
            [0, y, 0],
            [x, y, 0],
            [0, 0, z],
            [x, 0, z],
            [0, y, z],
            [x, y, z]
        ])
        world_point = self.local_to_world(all_points)
        # conservative AABB
        return world_point.min(axis = 0) - 0.1, world_point.max(axis = 0) + 0.1

    
    def __repr__(self):
        aabb_min, aabb_max = self.get_aabb()
        return f"<Volume grid {self.type_name.upper()} with phase {self.phase_type}, albedo: {self.albedo},\n \
                shape: (x = {self.xres}, y = {self.yres}, z = {self.zres}, c = {self.channel}),\n \
                AABB: Min = {aabb_min}, Max = {aabb_max}>"


@ti.func
def rgb_select(rgb: vec3, channel: int) -> float:
    """ Get the value of a specified channel 
        This implement looks dumb, but I think it has its purpose:
        Dynamic indexing will cause the local array get moved to global memory
        i.e. the access speed will drop significantly

        I am not sure whether Taichi Lang has the problem or not, yet I know CUDA
        has this problem
    """
    result = 0.0
    if channel == 0:
        result = rgb[0]
    elif channel == 1:
        result = rgb[1]
    else:
        result = rgb[2]
    return result

@ti.dataclass
class GridVolume:
    """ Grid Volume Taichi End definition"""
    _type:    int
    """ grid volume type: 0 (none), 1 (mono-chromatic), 2 (RGB) """
    albedo:   vec3            
    """ scattering albedo """
    inv_T:    mat3
    """ Inverse transform matrix: $(R_rotation @ R_scaling)^{-1}$ """
    trans:    vec3
    """ Translation vector (world frame) """
    mini:     vec3
    """ Min bound (AABB) minimum coordinate """
    maxi:     vec3
    """ Max bound (AABB) maximum coordinate """
    max_idxs: vec3i            # max_indices
    """ Maximum index for clamping, in case there's out-of-range access (xres - 1, yres - 1, zres - 1)"""
    majorant: vec3            # max_values
    """ (Scaled) Maximum value of sigma_t as majorant, for mono-chromatic, [0] is the value """
    pdf: vec3                   # normalized majorant
    """ PDF used for MIS (wavelength spectrum) """
    ph:     PhaseFunction   # phase function
    """ Phase function for scattering sampling """

    @ti.func
    def is_scattering(self):   # check whether the current medium is scattering medium
        return self._type >= 1

    @ti.func
    def intersect_volume(self, ray_o: vec3, ray_d: vec3, max_t: float) -> vec2:
        """ Get ray-volume intersection AABB near and far distance """
        near_far = vec2([0, max_t])
        inv_dir = 1.0 / ray_d

        t1s = (self.mini - ray_o) * inv_dir
        t2s = (self.maxi - ray_o) * inv_dir

        tmin = ti.min(t1s, t2s)
        tmax = ti.max(t1s, t2s)

        near_far[0] = ti.max(0, tmin.max()) + 1e-4
        near_far[1] = ti.min(max_t, tmax.min()) - 1e-4
        return near_far

    @ti.func
    def transmittance(self, grid: ti.template(), ray_o: vec3, ray_d: vec3, max_t: float) -> vec3:
        transm = vec3([1, 1, 1])
        if self._type:
            near_far = self.intersect_volume(ray_o, ray_d, max_t)
            if near_far[0] < near_far[1] and near_far[1] > 0:
                ray_o = self.inv_T @ (ray_o - self.trans)
                ray_d = self.inv_T @ ray_d
                if self._type:
                    if self._type == GridVolume_np.RGB:     # single channel
                        transm = self.eval_tr_ratio_tracking_3d(grid, ray_o, ray_d, near_far)
                    else:
                        transm = self.eval_tr_ratio_tracking_3d(grid, ray_o, ray_d, near_far)
        return transm
    
    @ti.func
    def sample_mfp(self, grid: ti.template(), ray_o: vec3, ray_d: vec3, max_t: float) -> vec4:
        transm = vec4([1, 1, 1, -1])
        if self._type: 
            near_far = self.intersect_volume(ray_o, ray_d, max_t)
            if near_far[0] < near_far[1] and near_far[1] > 0:
                ray_o = self.inv_T @ (ray_o - self.trans)
                ray_d = self.inv_T @ ray_d
                if self._type:
                    if self._type == 2:     # single channel
                        transm = self.sample_distance_delta_tracking_3d(grid, ray_o, ray_d, near_far)
                    else:
                        transm = self.sample_distance_delta_tracking_3d(grid, ray_o, ray_d, near_far)
        return transm
    
    @ti.func 
    def density_lookup_3d(self, grid: ti.template(), index: vec3, u_offset: vec3) -> vec3:
        """ Stochastic lookup of density (mono-chromatic volume) """
        idx = ti.cast(ti.floor(index + (u_offset - 0.5)), int)
        val = ZERO_V3
        if (idx >= 0).all() and (idx <= self.max_idxs).all():
            val = grid[idx[2], idx[1], idx[0]]
        return val
    
    @ti.func 
    def density_lookup(self, grid: ti.template(), index, u_offset: vec3) -> float:
        """ Stochastic lookup of density (RGB volume) """
        idx = ti.cast(ti.floor(index + (u_offset - 0.5)), int)
        val = 0.0
        if (idx >= 0).all() and (idx <= self.max_idxs).all():
            val = grid[idx[2], idx[1], idx[0]]
        # indexing pattern z, y, x
        return val
    
    @ti.func 
    def sample_distance_delta_tracking(self, grid: ti.template(), ray_ol: vec3, ray_dl: vec3, near_far: vec2) -> vec4:
        """ Sample distance (mono-chromatic volume) via delta tracking 
            Note that there is no 'sampling PDF', since we are not going to use it anyway
        """
        result = vec4([1, 1, 1, -1])
        if self._type:
            inv_maj = 1.0 / self.majorant[0]

            t = near_far[0]
            t -= ti.log(1.0 - ti.random(float)) * inv_maj
            while t < near_far[1]:
                d = self.density_lookup(grid, ray_ol + t * ray_dl, vec3([
                    ti.random(float), ti.random(float), ti.random(float)
                ]))
                # Scatter upon real collision
                if ti.random(float) < d * inv_maj:
                    result[:3]  = self.albedo
                    result[3]   = t 
                    break
                t -= ti.log(1.0 - ti.random(float)) * inv_maj
        return result

    @ti.func
    def eval_tr_ratio_tracking(self, grid: ti.template(), ray_ol: vec3, ray_dl: vec3, near_far: vec2) -> vec3:
        Tr = 1.0
        if self._type:
            inv_maj = 1.0 / self.majorant[0]

            t = near_far[0]
            while True:
                t -= ti.log(1.0 - ti.random(float)) * inv_maj
                if t >= near_far[1]: break
                d = self.density_lookup(grid, ray_ol + t * ray_dl, vec3([
                    ti.random(float), ti.random(float), ti.random(float)
                ]))
                Tr *= ti.max(0, 1.0 - d * inv_maj)
                # Russian Roulette
                if Tr < 0.1:
                    if ti.random(float) >= Tr:
                        Tr = 0.0
                        break
                    Tr = 1.0
        return vec3([Tr, Tr, Tr])
    
    @ti.func
    def sample_new_rays(self, incid: vec3):
        ret_spec = vec3([1, 1, 1])
        ret_dir  = incid
        ret_pdf  = 1.0
        if self.is_scattering():   # medium interaction - evaluate phase function (currently output a float)
            local_new_dir, ret_pdf = self.ph.sample_p(incid)     # local frame ray_dir should be transformed
            ret_dir, _ = delocalize_rotate(incid, local_new_dir)
            ret_spec *= ret_pdf
        return ret_dir, ret_spec, ret_pdf
    
    @ti.func 
    def sample_distance_delta_tracking_3d(self, grid: ti.template(), ray_ol: vec3, ray_dl: vec3, near_far: vec2) -> vec4:
        result = vec4([1, 1, 1, -1])
        if self._type:
            # mis_pdf = ONES_V3
            
            # Step 1: choose one element according to the majorant
            channel = 2
            maj     = self.majorant[2]
            val     = ti.random(float)
            if val < self.pdf[0]:
                maj     = self.majorant[0]
                channel = 0
            elif val < self.pdf[0] + self.pdf[1]:
                maj     = self.majorant[1]
                channel = 1
            inv_maj     = 1.0 / maj

            t = near_far[0]
            dist = -ti.log(1.0 - ti.random(float)) * inv_maj
            while t + dist < near_far[1]:
                t += dist

                d = self.density_lookup_3d(grid, ray_ol + t * ray_dl, vec3([
                    ti.random(float), ti.random(float), ti.random(float)
                ]))

                # Scatter upon real collision
                n_s = rgb_select(d, channel)
                if ti.random(float) < n_s * inv_maj:
                    result[:3] *= self.albedo
                    result[3]   = t 
                    # mis_pdf *= d / ti.max(EPSL_V3, self.majorant)
                    break
                
                maj_tr      = ti.exp(-self.majorant * dist)
                result[:3] *= maj_tr / ti.exp(- dist * maj)
                # mis_pdf    *= maj_tr
                dist        = -ti.log(1.0 - ti.random(float)) * inv_maj
            if t + dist >= near_far[1]:
                dist = near_far[1] - t

            maj_tr      = ti.exp(-self.majorant * dist)
            result[:3] *= maj_tr / ti.exp(- dist * maj)
            # mis_pdf    *= maj_tr

            # MIS
            # mis_numerator = rgb_select(mis_pdf, channel)
            # result[:3]   *= mis_numerator / mis_pdf.sum()
        return result
    
    @ti.func
    def eval_tr_ratio_tracking_3d(self, grid: ti.template(), ray_ol: vec3, ray_dl: vec3, near_far: vec2) -> vec3:
        Tr = ONES_V3
        if self._type:
            # mis_pdf = ONES_V3

            # Step 1: choose one element according to the majorant
            channel = 2
            maj     = self.majorant[2]
            val     = ti.random(float)
            if val < self.pdf[0]:
                maj     = self.majorant[0]
                channel = 0
            elif val < self.pdf[0] + self.pdf[1]:
                maj     = self.majorant[1]
                channel = 1
            inv_maj     = 1.0 / maj
            
            t = near_far[0]
            while True:
                dist = -ti.log(1.0 - ti.random(float)) * inv_maj
                
                if t + dist >= near_far[1]: 
                    dist = near_far[1] - t      # get the actual optical length
                    Tr *= ti.exp(- self.majorant * dist) / ti.exp(- dist * maj)
                    break
                t += dist
                # for mono-chromatic medium, this is 1
                d = self.density_lookup_3d(grid, ray_ol + t * ray_dl, vec3([
                    ti.random(float), ti.random(float), ti.random(float)
                ]))

                majorant_tr = ti.exp(- self.majorant * dist)
                Tr *= ti.max(ZERO_V3, 1.0 - d * inv_maj) * majorant_tr / ti.exp(- dist * maj)
                # mis_pdf *= d * majorant_tr * self.pdf

                # Russian Roulette
                if Tr.max() < 0.1:
                    rr_max_val = Tr.max()
                    if ti.random(float) >= rr_max_val:
                        Tr.fill(0)
                        break
                    Tr /= rr_max_val
        
            # evaluate MIS weight
            # mis_numerator = rgb_select(mis_pdf, channel)
            # Tr *= mis_numerator / mis_pdf.sum()
        return Tr
    