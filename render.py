"""
    Rendering main executable
    @author: Qianyue He
    @date: 2023.2.7
"""
import os
import taichi as ti
from tqdm import tqdm
from taichi.math import vec3

from typing import List
from la.cam_transform import *
from tracer.path_tracer import PathTracer
from emitters.abtract_source import LightSource

from scene.obj_desc import ObjDescriptor
from scene.xml_parser import mitsuba_parsing
from scene.opts import get_options, mapped_arch
from sampler.general_sampling import mis_weight
from utils.tools import folder_path, TicToc
from utils.watermark import apply_watermark

@ti.data_oriented
class Renderer(PathTracer):
    """
        Simple Ray tracing using Bary-centric coordinates
        This tracer can yield result with global illumination effect
    """
    def __init__(self, emitters: List[LightSource], objects: List[ObjDescriptor], prop: dict):
        super().__init__(emitters, objects, prop)
        self.clock = TicToc()
        
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
                        mis_w = 1.0
                        if not emitter.is_delta:
                            bsdf_pdf = self.get_pdf(obj_id, light_dir, normal, ray_d, self.world.medium)
                            mis_w    = mis_weight(light_pdf, bsdf_pdf)
                        # FIXME: here we have a bug
                        direct_int  += direct_spec * shadow_int * mis_w / emitter_pdf
                    else:
                        direct_int += direct_spec * shadow_int / emitter_pdf
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
    
    def summary(self):
        print(f"[INFO] Finished rendering. SPP = {self.cnt[None]}. Rendering time: {self.clock.toc():.3f} s")

if __name__ == "__main__":
    from utils.tools import TicToc
    options = get_options()
    cache_path = folder_path(f"./cached/{options.scene}", f"Cache path for scene {options.scene} not found. JIT compilation might take some time (~30s)...")
    ti.init(arch = mapped_arch(options.arch), kernel_profiler = options.profile, \
            default_ip = ti.i32, default_fp = ti.f32, offline_cache_file_path = cache_path)
    input_folder = os.path.join(options.input_path, options.scene)
    emitter_configs, _, meshes, configs = mitsuba_parsing(input_folder, options.name)  # complex_cornell
    rdr = Renderer(emitter_configs, meshes, configs)
    gui = ti.GUI('Path Tracing', (rdr.w, rdr.h))
    
    max_iter_num = options.iter_num if options.iter_num > 0 else 10000
    iter_cnt = 0
    print("[INFO] starting to loop...")
    for iter_cnt in tqdm(range(max_iter_num)):
        gui.clear()
        rdr.reset()
        for e in gui.get_events(gui.PRESS):
            if e.key == gui.ESCAPE:
                gui.running = False
        rdr.render()
        gui.set_image(rdr.pixels)
        gui.show()
        if gui.running == False: break
    rdr.summary()
    if options.profile:
        ti.profiler.print_kernel_profiler_info() 
    image = apply_watermark(rdr.pixels)
    ti.tools.imwrite(image, f"{folder_path(options.output_path)}{options.img_name}-{options.name[:-4]}.{options.img_ext}")
