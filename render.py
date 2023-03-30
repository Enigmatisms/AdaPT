"""
    Rendering main executable
    @author: Qianyue He
    @date: 2023.2.7
"""
import os
import numpy as np
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
from utils.tdom_analyze import time_domain_curve
from scene.xml_parser import mitsuba_parsing
from scene.opts import get_options, mapped_arch

rdr_mapping = {"pt": Renderer, "vpt": VolumeRenderer, "bdpt": BDPT}
name_mapping = {"pt": "", "vpt": "Volumetric ", "bdpt": "Bidirectional "}

def export_transient_profile(
    rdr: BDPT, sample_cnt: int, out_path: str, out_name: str, out_ext: str, 
    normalize: float = 0., save_trans: bool = True, analyze: bool = False
):
    output_folder = folder_path(os.path.join(out_path, out_name))
    all_files = []
    print(f"[INFO] Transient profile post processing... ")
    for i in tqdm(range(sample_cnt)):
        rdr.copy_average(i)
        transient_img = apply_watermark(rdr, 0.0, False)
        all_files.append(transient_img)
    all_files = np.stack(all_files, axis = 0)
    print(f"[INFO] Exporting transient profile to folder '{output_folder}'")
    if normalize > 0.9:
        qnt = np.quantile(all_files, normalize)
        for i in tqdm(range(sample_cnt)):
            all_files[i, ...] /= qnt
    if save_trans:
        for i in tqdm(range(sample_cnt)):
            ti.tools.imwrite(all_files[i, ...], f"{output_folder}/img_{i + 1:03d}.{out_ext}")
    if analyze:
        print(f"[INFO] Analyzing time domain information...")
        time_domain_curve(all_files, name = out_name, viz = False)

if __name__ == "__main__":
    opts = get_options()
    cache_path = folder_path(f"./cached/{opts.scene}", f"[INFO] Cache path for scene {opts.scene} not found. JIT compilation might take some time (~30s)...")
    ti.init(arch = mapped_arch(opts.arch), kernel_profiler = opts.profile, device_memory_fraction = 0.8, offline_cache = not opts.no_cache, \
            default_ip = ti.i32, default_fp = ti.f32, offline_cache_file_path = None if opts.no_cache else cache_path, debug = opts.debug)
    input_folder = os.path.join(opts.input_path, opts.scene)
    emitter_configs, _, meshes, configs = mitsuba_parsing(input_folder, opts.name)  # complex_cornell
    output_folder = f"{folder_path(opts.output_path)}"
    output_freq = opts.output_freq
    if output_freq > 0:
        output_folder = folder_path(f"{output_folder}{opts.img_name}-{opts.name[:-4]}-{opts.type}/")
    rdr: PathTracer = rdr_mapping[opts.type](emitter_configs, meshes, configs)
    if type(rdr) != BDPT and configs.get('decomposition', 'none').startswith('transient'):
        print("[Warning] Transient rendering is only supported in BDPT renderer.")

    max_iter_num = opts.iter_num if opts.iter_num > 0 else configs.get('iter_num', 2000)
    iter_cnt = 0
    
    eye_start = configs.get('start_t', 1)
    lit_start = configs.get('start_s', 0)
    max_bounce = configs.get('max_bounce', 16)
    max_depth = configs.get('max_depth', 16)
    eye_end = configs.get('end_t', max_bounce + 2)      # one more vertex for each path (starting vertex)
    lit_end = configs.get('end_s', max_bounce + 2)      # one more vertex for each path (starting vertex)
    print(f"[INFO] starting to loop. Max eye vnum: {eye_end}, max light vnum: {lit_end}.")
    if opts.type == "bdpt":
        print(f"[INFO] BDPT with {max_bounce} bounce(s), max depth: {max_depth}. Cam vertices [{eye_start}, {eye_end}], light vertices [{lit_start}, {lit_end}]")
    else:
        print(f"[INFO] {name_mapping[opts.type]}Path Tracing with {max_bounce} bounce(s)")

    if opts.no_gui:
        try:
            for iter_cnt in tqdm(range(max_iter_num)):
                rdr.render(eye_start, eye_end, lit_start, lit_end, max_bounce, max_depth)
        except KeyboardInterrupt:
            print("[QUIT] Quit on Keyboard interruptions")
    else:
        window   = tui.Window(f"{name_mapping[opts.type]}Path Tracing", res = (rdr.w, rdr.h))
        canvas = window.get_canvas()
        gui = window.get_gui()
        for iter_cnt in tqdm(range(max_iter_num)):
            rdr.reset()
            for e in window.get_events(tui.PRESS):
                if e.key == tui.ESCAPE:
                    window.running = False
            rdr.render(eye_start, eye_end, lit_start, lit_end, max_bounce, max_depth)
            canvas.set_image(rdr.pixels)
            window.show()
            if window.running == False: break
            if output_freq > 0 and iter_cnt % output_freq == 0:
                image = rdr.pixels.to_numpy()
                ti.tools.imwrite(image, f"{output_folder}img_{iter_cnt:05d}.{opts.img_ext}")
    rdr.summary()
    if opts.profile:
        ti.profiler.print_kernel_profiler_info() 
        ti.profiler.memory_profiler.print_memory_profiler_info()
    image = apply_watermark(rdr, opts.normalize, True, not opts.no_watermark)
    save_figure = not opts.no_save_fig
    if save_figure:
        ti.tools.imwrite(image, f"{folder_path(opts.output_path)}{opts.img_name}-{opts.name[:-4]}-{opts.type}.{opts.img_ext}")
    if type(rdr) == BDPT and rdr.decomp[None] > 0:
        export_transient_profile(rdr, configs['sample_count'], opts.output_path, opts.name[:-4], opts.img_ext, opts.normalize, save_figure, opts.analyze)
