"""
    Rendering main executable
    @author: Qianyue He
    @date: 2023.2.7
"""
import os
import taichi as ti
import taichi.ui as tui
from tqdm import tqdm

from renderer.bdpt import BDPT
from renderer.vpt import VolumeRenderer
from renderer.vanilla_renderer import Renderer
from tracer.path_tracer import PathTracer
from la.cam_transform import *

from utils.tools import folder_path
from utils.watermark import apply_watermark
from scene.xml_parser import mitsuba_parsing
from scene.opts import get_options, mapped_arch

rdr_mapping = {"pt": Renderer, "vpt": VolumeRenderer, "bdpt": BDPT}
name_mapping = {"pt": "", "vpt": "Volumetric ", "bdpt": "Bidirectional "}

if __name__ == "__main__":
    options = get_options()
    cache_path = folder_path(f"./cached/{options.scene}", f"Cache path for scene {options.scene} not found. JIT compilation might take some time (~30s)...")
    ti.init(arch = mapped_arch(options.arch), kernel_profiler = options.profile, \
            default_ip = ti.i32, default_fp = ti.f32, offline_cache_file_path = cache_path, debug = options.debug)
    input_folder = os.path.join(options.input_path, options.scene)
    emitter_configs, _, meshes, configs = mitsuba_parsing(input_folder, options.name)  # complex_cornell

    rdr: PathTracer = rdr_mapping[options.type](emitter_configs, meshes, configs)

    max_iter_num = options.iter_num if options.iter_num > 0 else 10000
    iter_cnt = 0
    print("[INFO] starting to loop...")
    if options.no_gui:
        try:
            for iter_cnt in tqdm(range(max_iter_num)):
                rdr.render()
        except KeyboardInterrupt:
            print("[QUIT] Quit on Keyboard interruptions")
    else:
        window   = tui.Window(f"{name_mapping[options.type]}Path Tracing", res = (rdr.w, rdr.h))
        canvas = window.get_canvas()
        gui = window.get_gui()
        for iter_cnt in tqdm(range(max_iter_num)):
            rdr.reset()
            for e in window.get_events(tui.PRESS):
                if e.key == tui.ESCAPE:
                    window.running = False
            rdr.render()
            canvas.set_image(rdr.pixels)
            window.show()
            if window.running == False: break
    rdr.summary()
    if options.profile:
        ti.profiler.print_kernel_profiler_info() 
    image = apply_watermark(rdr.pixels, True)
    ti.tools.imwrite(image, f"{folder_path(options.output_path)}{options.img_name}-{options.name[:-4]}.{options.img_ext}")
