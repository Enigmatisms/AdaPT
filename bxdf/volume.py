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
from renderer.constants import ZERO_V3, ONES_V3
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

class GridVolume_np:
    NONE = 0
    MONO = 1
    RGB  = 2
    __type_mapping = {"none": NONE, "mono": MONO, "rgb": RGB}
    def __init__(self, elem: xet.Element):
        self.albedo = np.ones(3, np.float32)
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
        self.mono2rgb = False

        elem_to_query = {
            "rgb": rgb_parse, 
            "float": lambda el: get(el, "value"), 
            "string": lambda el: self.setup_volume(get(el, "path", str)),
            "transform": lambda el: self.assign_transform(*transform_parse(el)),
            "bool": lambda el: el.get("value") in {"True", "true"}
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

            if self.mono2rgb == True:
                if self.channel == 3:
                    CONSOLE.log("Setting 'mono2rgb = True' is meanless for RGB volume. This setting is only meaningful when the volume is mono-chromatic.")
                else:
                    CONSOLE.log("Mono-chromatic volume is converted to RGB volume.")
                    self.channel = 3
                    self.type_id = GridVolume_np.RGB
                    self.density_grid = GridVolume_np.make_colorful_volume(self.density_grid, self.xres, self.yres, self.zres)
            else:
                self.density_grid = np.concatenate([self.density_grid, self.density_grid, self.density_grid], axis = -1)
            
            if self.scale is None:
                self.scale = np.eye(3, dtype = np.float32)
            else:
                self.scale = np.diag(self.scale)
            self.forward_t = self.rotation @ self.scale
            if self.type_id == GridVolume_np.MONO:
                self.density_grid *= self.density_scaling[0]
                CONSOLE.log(f"Volume density scaled by {self.density_scaling[0]}.")
            else:
                self.density_grid *= self.density_scaling
                CONSOLE.log(f"Volume density scaled by {self.density_scaling}.")

    def assign_transform(self, trans_r, trans_t, trans_s):
        self.rotation = trans_r
        self.offset   = trans_t
        self.scale    = trans_s

    def setup_volume(self, path:str):
        if not os.path.exists(path):
            CONSOLE.log(f"[red]Error :skull:[/red]: {path} contains no valid volume file.")
            raise RuntimeError("Volume file not found.")
        density_grid, (self.xres, self.yres, self.zres, self.channel) = vol_file_to_numpy(path, FORCE_MONO_COLOR)
        if FORCE_MONO_COLOR:
            CONSOLE.log(f"[yellow]Warning[/yellow]: FORCE_MONO_COLOR is True. This only makes sense when we are testing the code.")

        density_grid = density_grid.reshape((self.zres, self.yres, self.xres, self.channel))
        return density_grid
    
    @staticmethod
    def make_colorful_volume(density_grid: np.ndarray, xres: int, yres: int, zres: int):
        z_coords = np.linspace(0, 0.9, zres, dtype = np.float32).reshape(-1, 1, 1, 1) + 0.1
        y_coords = np.linspace(0, 0.9, yres, dtype = np.float32).reshape(1, -1, 1, 1) + 0.1
        x_coords = np.linspace(0, 0.9, xres, dtype = np.float32).reshape(1, 1, -1, 1) + 0.1
        density_grid = np.concatenate([
            density_grid,
            density_grid,
            density_grid,
        ], axis = -1)
        return density_grid
    
    def get_shape(self) -> Tuple[int, int, int]:
        return (self.zres, self.yres, self.xres)
    
    def get_majorant(self, guard = 0.2, scale_ratio = 1.05):
        maj       = self.density_grid.max(axis = (0, 1, 2))
        maj_guard = np.mean(maj) * guard
        maj       = np.maximum(maj, maj_guard)
        maj      *= scale_ratio
        return vec3(maj) if self.type_id == GridVolume_np.RGB else vec3([maj, maj, maj])   

    def export(self):
        if self.type_id == GridVolume_np.NONE:
            return GridVolume(_type = 0)
        aabb_mini, aabb_maxi = self.get_aabb()
        majorant = self.get_majorant()
        CONSOLE.log(f"Majorant: {majorant}. PDF: {majorant / majorant.sum()}")      
        # the shape of density grid: (zres, yres, xres, channels)
        return GridVolume(
            _type   = self.type_id,
            albedo  = vec3(self.albedo),
            inv_T   = mat3(np.linalg.inv(self.forward_t)),
            trans   = vec3(self.offset),
            mini    = vec3(aabb_mini), 
            maxi    = vec3(aabb_maxi),
            max_idxs = vec3i([self.xres - 1, self.yres - 1, self.zres - 1]),
            majorant = majorant,
            pdf      = majorant / majorant.sum(),
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
        return world_point.min(axis = 0) - 0.01, world_point.max(axis = 0) + 0.01

    
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

        near_far[0] = ti.max(0, tmin.max()) + 1e-5
        near_far[1] = ti.min(max_t, tmax.min()) - 1e-5
        return near_far

    @ti.func
    def transmittance(self, grid: ti.template(), ray_o: vec3, ray_d: vec3, thp: vec3, max_t: float) -> vec3:
        transm = vec3([1, 1, 1])
        if self._type:
            near_far = self.intersect_volume(ray_o, ray_d, max_t)
            if near_far[0] < near_far[1] and near_far[1] > 0:
                ray_o = self.inv_T @ (ray_o - self.trans)
                ray_d = self.inv_T @ ray_d
                if self._type:
                    transm = self.eval_tr_ratio_tracking_3d(grid, ray_o, ray_d, thp, near_far)
        return transm
    
    @ti.func
    def sample_mfp(self, grid: ti.template(), ray_o: vec3, ray_d: vec3, thp: vec3, max_t: float) -> vec4:
        transm = vec4([1, 1, 1, -1])
        if self._type: 
            near_far = self.intersect_volume(ray_o, ray_d, max_t)
            if near_far[0] < near_far[1] and near_far[1] > 0:
                ray_o = self.inv_T @ (ray_o - self.trans)
                ray_d = self.inv_T @ ray_d
                if self._type:
                    transm = self.sample_distance_delta_tracking_3d(grid, ray_o, ray_d, thp, near_far)
        return transm
    
    @ti.func 
    def density_lookup_3d(self, grid: ti.template(), index: vec3, u_offset: vec3) -> vec3:
        """ Stochastic lookup of density (mono-chromatic volume) """
        idx   = ti.cast(ti.floor(index + (u_offset - 0.5)), int)
        val   = ZERO_V3
        if (idx >= 0).all() and (idx <= self.max_idxs).all():
            val   = grid[idx[2], idx[1], idx[0]]
        return val
    
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
    def sample_distance_delta_tracking_3d(self, 
        grid: ti.template(), ray_ol: vec3, ray_dl: vec3, thp: vec3, near_far: vec2
    ) -> vec4:
        result = vec4([1, 1, 1, -1])
        if self._type:
            # Step 1: choose one element according to the majorant
            pdfs    = thp * self.pdf
            pdfs   /= pdfs.sum()
            Tr      = 1.0
            pdf     = 1.0
            albedo  = 0.0
            maj     = 0.0
            channel = 0
            val     = ti.random(float)

            # avoid dynamic indexing on GPU (and array might be moved from local registers to global memory)
            if val <= pdfs[0]:
                albedo  = self.albedo[0]
                maj     = self.majorant[0]
                pdf     = pdfs[0]
                channel = 0
            elif val <= pdfs[0] + pdfs[1]:
                albedo  = self.albedo[1]
                maj     = self.majorant[1]
                pdf     = pdfs[1]
                channel = 1
            else:
                albedo  = self.albedo[2]
                maj     = self.majorant[2]
                pdf     = pdfs[2]
                channel = 2
            inv_maj     = 1.0 / maj

            t = near_far[0] - ti.log(1.0 - ti.random(float)) * inv_maj
            while t < near_far[1]:
                d = self.density_lookup_3d(grid, ray_ol + t * ray_dl, vec3([
                    ti.random(float), ti.random(float), ti.random(float)
                ]))
                # Scatter upon real collision
                n_s = rgb_select(d, channel)
                if ti.random(float) < n_s * inv_maj:
                    Tr       *= albedo
                    result[3] = t 
                    break
                
                t -= ti.log(1.0 - ti.random(float)) * inv_maj
            if self._type == GridVolume_np.RGB:
                if channel == 0:
                    result[:3] = vec3([Tr / pdf, 0, 0])
                elif channel == 1:
                    result[:3] = vec3([0, Tr / pdf, 0])
                else:
                    result[:3] = vec3([0, 0, Tr / pdf])
            else:
                result[:3] = vec3([Tr, Tr, Tr])
        return result
    
    @ti.func
    def eval_tr_ratio_tracking_3d(self, 
        grid: ti.template(), ray_ol: vec3, ray_dl: vec3, thp: vec3, near_far: vec2
    ) -> vec3:
        transm = ONES_V3
        if self._type:
            # Step 1: choose one element according to the majorant
            pdfs    = thp * self.pdf
            pdfs   /= pdfs.sum()
            Tr      = 1.0
            pdf     = 1.0
            maj     = 0.0
            channel = 0
            val     = ti.random(float)

            # avoid dynamic indexing on GPU (and array might be moved from local registers to global memory)
            if val <= pdfs[0]:
                maj     = self.majorant[0]
                pdf     = pdfs[0]
                channel = 0
            elif val <= pdfs[0] + pdfs[1]:
                maj     = self.majorant[1]
                pdf     = pdfs[1]
                channel = 1
            else:
                maj     = self.majorant[2]
                pdf     = pdfs[2]
                channel = 2
            inv_maj     = 1.0 / maj
            
            t = near_far[0]
            while True:
                # problem with coordinates
                t -= ti.log(1.0 - ti.random(float)) * inv_maj
                
                if t >= near_far[1]: break
                # for mono-chromatic medium, this is 1
                d = self.density_lookup_3d(grid, ray_ol + t * ray_dl, vec3([
                    ti.random(float), ti.random(float), ti.random(float)
                ]))

                n_s = rgb_select(d, channel)
                Tr *= ti.max(0.0, 1.0 - n_s * inv_maj)

                # Russian Roulette
                if Tr < 0.1:
                    if ti.random(float) >= Tr:
                        Tr = 0.0
                        break
                    Tr = 1.0
            if self._type == GridVolume_np.RGB:
                if channel == 0:
                    transm = vec3([Tr / pdf, 0, 0])
                elif channel == 1:
                    transm = vec3([0, Tr / pdf, 0])
                else:
                    transm = vec3([0, 0, Tr / pdf])
            else:
                transm = vec3([Tr, Tr, Tr])
        return transm
    