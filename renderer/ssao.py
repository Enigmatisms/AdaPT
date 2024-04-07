""" Screen-Space Ambient Occlusion (SSAO)
    @author: Qianyue He
    @date:   2024-4-7
"""
import taichi as ti
from taichi.math import vec3

from typing import List
from la.cam_transform import *
from tracer.path_tracer import PathTracer
from emitters.abtract_source import LightSource
from renderer.constants import ZERO_V3

from parsers.obj_desc import ObjDescriptor
from sampler.general_sampling import uniform_hemisphere

from rich.console import Console
CONSOLE = Console(width = 128)

@ti.data_oriented
class SSAORenderer(PathTracer):
    """
        Renderer Final Class
    """
    def __init__(self, 
        emitters: List[LightSource], array_info: dict, 
        objects: List[ObjDescriptor], prop: dict
    ):
        super().__init__(emitters, array_info, objects, prop)
        self.smp_hemisphere = prop.get('smp_hemisphere', 32)
        self.depth_samples  = prop.get('depth_samples', 64)
        self.sample_extent  = prop.get('sample_extent', 0.1)         # float
        CONSOLE.log(f"Rendering depth map: {self.depth_samples} sample(s) per pixel.")
        self.get_depth_map()
        CONSOLE.log(f"Depth map rendering completed.")
        
    @ti.kernel
    def get_depth_map(self):
        ti.loop_config(parallelize = 8, block_dim = 512)
        # first get 
        for i, j in self.pixels:
            in_crop_range = i >= self.start_x and i < self.end_x and j >= self.start_y and j < self.end_y
            if not self.do_crop or in_crop_range:
                num_valid_hits = 0
                for _ in range(self.depth_samples):
                    ray_d = self.pix2ray(i, j)
                    ray_o = self.cam_t
                    it = self.ray_intersect(ray_d, ray_o)
                    if it.is_ray_not_hit(): continue 
                    num_valid_hits += 1
                    self.color[i, j][2] += it.min_depth
                
                if num_valid_hits:
                    self.color[i, j][2] /= num_valid_hits
                    
    @ti.func
    def get_sample_depth(self, it: ti.template(), pos: vec3):
        """ Get samples around normal 
            and return the screen space depth
        """
        local_dir = uniform_hemisphere()
        normal_sample, _ = delocalize_rotate(it.n_s, local_dir)
        position = pos + normal_sample * self.sample_extent
        depth = (position - self.cam_t).norm() 
        return depth
        
    @ti.kernel
    def render(self, _t_start: int, _t_end: int, _s_start: int, _s_end: int, _a: int, _b: int):
        self.cnt[None] += 1
        ti.loop_config(parallelize = 8, block_dim = 512)
        
        for i, j in self.pixels:
            in_crop_range = i >= self.start_x and i < self.end_x and j >= self.start_y and j < self.end_y
            if not self.do_crop or in_crop_range:
                min_depth = self.color[i, j][2] + 1e-5
                ray_d = self.pix2ray(i, j)
                ray_o = self.cam_t
                it = self.ray_intersect(ray_d, ray_o)
                if it.is_ray_not_hit(): break 
                # AO sampling: the hemisphere
                pos = ray_o + ray_d * it.min_depth
                num_un_occluded = 0.0
                for _ in range(self.smp_hemisphere):
                    depth = self.get_sample_depth(it, pos)
                    num_un_occluded += float(depth < min_depth)            # depth
                self.color[i, j][0] += num_un_occluded / self.smp_hemisphere
            color_vec = ZERO_V3
            color_vec.fill(self.color[i, j][2] / self.cnt[None])
            self.pixels[i, j] = color_vec
    
    def summary(self):
        super().summary()
        CONSOLE.print(f"SSAO SPP = {self.cnt[None]}. Rendering time: {self.clock.toc():.3f} s", justify="center")
