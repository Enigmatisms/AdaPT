""" Screen-Space Ambient Occlusion (SSAO)
    @author: Qianyue He
    @date:   2024-4-7
"""
import taichi as ti
import taichi.math as tm
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

@ti.func
def smooth_step(edge0: float, edge1: float, x: float) -> float:
    t = tm.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t)

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
        self.inv_cam_r      = self.cam_r.inverse()
        self.cam_normal     = (self.cam_r @ vec3([0, 0, 1])).normalized()
        CONSOLE.log(f"Rendering depth map: {self.depth_samples} sample(s) per pixel.")
        self.get_depth_map()
        CONSOLE.log(f"Depth map rendering completed.")
        CONSOLE.log(f"SSAO statistics: Sample per hemisphere: {self.smp_hemisphere} | Sample extent: {self.sample_extent:.2f}")
        
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
    def rasterize_pinhole(self, local_ray_x: float, local_ray_y: float):
        """ For path with only one camera vertex, ray should be re-rasterized to the film
            ray_d is pointing into the camera, therefore should be negated
        """
        valid_raster = False

        pi = int(self.half_w + 1.0 - local_ray_x / self.inv_focal)
        pj = int(self.half_h + 1.0 + local_ray_y / self.inv_focal)
        if pi >= self.start_x and pj >= self.start_y and pi < self.end_x and pj < self.end_y:   # cropping is considered
            valid_raster = True
        return pj, pi, valid_raster
    
    @ti.func
    def splat_camera(self, ray_d: vec3):
        """ Rasterize pos onto the image plane and query the depth map
        """
        test_depth = 0.0
        if tm.dot(ray_d, self.cam_normal) > 0.:
            local_ray = self.inv_cam_r @ ray_d
            z = local_ray[2]
            if z > 0.0:
                local_ray /= z
                p_row, p_col, is_valid = self.rasterize_pinhole(local_ray[0], local_ray[1])
                if is_valid:
                    test_depth = self.color[p_col, p_row][2]
        return test_depth
                    
    @ti.func
    def normal_sample_occluded(self, it: ti.template(), pos: vec3):
        """ Get samples around normal 
            and return the screen space depth
        """
        local_dir, _pdf = uniform_hemisphere()
        normal_sample, _r = delocalize_rotate(it.n_s, local_dir)
        position = pos + normal_sample * self.sample_extent
        # should rasterize the position on to the camera 
        ray_d         = position - self.cam_t
        ray_d        /= ray_d.norm()
        ## Online quering
        # it            = self.ray_intersect(ray_d, self.cam_t)
        # queried_depth = it.min_depth
        queried_depth = self.splat_camera(ray_d) + 1e-3
        depth         = (position - self.cam_t).norm()
        return ti.select(depth >= queried_depth, 1.0, 0.0) * smooth_step(0.0, 1.0, self.sample_extent / ti.abs(queried_depth - depth))
        
    @ti.kernel
    def render(self, _t_start: int, _t_end: int, _s_start: int, _s_end: int, _a: int, _b: int):
        self.cnt[None] += 1
        ti.loop_config(parallelize = 8, block_dim = 512)
        
        for i, j in self.pixels:
            in_crop_range = i >= self.start_x and i < self.end_x and j >= self.start_y and j < self.end_y
            color_vec = ZERO_V3
            if not self.do_crop or in_crop_range:
                ray_d = self.pix2ray(i, j)
                it = self.ray_intersect(ray_d, self.cam_t)
                if not it.is_ray_not_hit():  
                    # AO sampling: the hemisphere
                    pos = self.cam_t + ray_d * it.min_depth
                    occlusion_factor = 0.0
                    for _ in range(self.smp_hemisphere):
                        occlusion_factor += self.normal_sample_occluded(it, pos)
                    self.color[i, j][0] += 1.0 - occlusion_factor / self.smp_hemisphere
                    color_vec.fill(self.color[i, j][0] / self.cnt[None] )
            self.pixels[i, j] = color_vec
    
    def summary(self):
        super().summary()
        CONSOLE.print(f"SSAO SPP = {self.cnt[None]}. Rendering time: {self.clock.toc():.3f} s", justify="center")
