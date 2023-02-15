"""
    Rendering main executable
    @author: Qianyue He
    @date: 2023.2.7
"""
import os
import taichi as ti
from tqdm import tqdm

from la.cam_transform import *
from renderer.vanilla_renderer import Renderer
from renderer.vpt import VolumeRenderer

from utils.tools import folder_path
from utils.watermark import apply_watermark
from scene.xml_parser import mitsuba_parsing
from scene.opts import get_options, mapped_arch

if __name__ == "__main__":
    options = get_options()
    cache_path = folder_path(f"./cached/{options.scene}", f"Cache path for scene {options.scene} not found. JIT compilation might take some time (~30s)...")
    ti.init(arch = mapped_arch(options.arch), kernel_profiler = options.profile, \
            default_ip = ti.i32, default_fp = ti.f32, offline_cache_file_path = cache_path)
    input_folder = os.path.join(options.input_path, options.scene)
    emitter_configs, _, meshes, configs = mitsuba_parsing(input_folder, options.name)  # complex_cornell
    if options.vanilla:
        rdr = Renderer(emitter_configs, meshes, configs)
    else:
        rdr = VolumeRenderer(emitter_configs, meshes, configs)
    gui = ti.GUI(f"{'' if options.vanilla else 'Volumetric '}Path Tracing", (rdr.w, rdr.h))
    
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
    image = apply_watermark(rdr.pixels, True)
    ti.tools.imwrite(image, f"{folder_path(options.output_path)}{options.img_name}-{options.name[:-4]}.{options.img_ext}")
