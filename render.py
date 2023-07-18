"""
    Rendering main executable
    @author: Qianyue He
    @date: 2023.2.7
"""
import os
import pickle
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
from parsers.xml_parser import scene_parsing
from parsers.opts import get_options, mapped_arch

from utils.rich_utils import ItersPerSecColumn
from rich.progress import (BarColumn, Progress, MofNCompleteColumn, TextColumn, SpinnerColumn, 
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.console import Console

CONSOLE = Console(width = 128)

rdr_mapping = {"pt": Renderer, "vpt": VolumeRenderer, "bdpt": BDPT}
name_mapping = {"pt": "", "vpt": "Volumetric ", "bdpt": "Bidirectional "}

def export_transient_profile(
    rdr: BDPT, sample_cnt: int, out_path: str, out_name: str, out_ext: str, 
    normalize: float = 0., save_trans: bool = True, analyze: bool = False
):
    output_folder = folder_path(os.path.join(out_path, out_name))
    all_files = []
    CONSOLE.log(f"Transient profile post processing... ")
    for i in tqdm(range(sample_cnt)):
        rdr.copy_average(i)
        transient_img = apply_watermark(rdr, 0.0, False)
        all_files.append(transient_img)
    all_files = np.stack(all_files, axis = 0)
    CONSOLE.log(f":floppy_disk: Exporting transient profile to folder [green]'{output_folder}'")
    if normalize > 0.9:
        qnt = np.quantile(all_files, normalize)
        for i in tqdm(range(sample_cnt)):
            all_files[i, ...] /= qnt
    if save_trans:
        for i in tqdm(range(sample_cnt)):
            ti.tools.imwrite(all_files[i, ...], f"{output_folder}/img_{i + 1:03d}.{out_ext}")
    if analyze:
        CONSOLE.log(f":minidisc: Analyzing time domain information... :minidisc:")
        time_domain_curve(all_files, name = out_name, viz = False)

def save_check_point(chkpt: dict, opts):
    chkpt_path = os.path.join(folder_path(opts.chkpt_path), f"{opts.img_name}-{opts.name[:-4]}-{opts.type}.pkl")
    with open(chkpt_path, 'wb') as file:
        pickle.dump(chkpt, file, protocol = pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    opts = get_options()
    cache_path = folder_path(f"./cached/{opts.scene}", f"Cache path for scene {opts.scene} not found. JIT compilation might take some time (~30s)...")
    ti.init(arch = mapped_arch(opts.arch), kernel_profiler = opts.profile, device_memory_fraction = 0.8, offline_cache = not opts.no_cache, \
            default_ip = ti.i32, default_fp = ti.f32, offline_cache_file_path = cache_path, debug = opts.debug)
    input_folder = os.path.join(opts.input_path, opts.scene)
    emitter_configs, array_info, all_objs, configs = scene_parsing(input_folder, opts.name)  # complex_cornell
    output_folder = f"{folder_path(opts.output_path)}"
    output_freq = opts.output_freq
    if output_freq > 0:
        output_folder = folder_path(f"{output_folder}{opts.img_name}-{opts.name[:-4]}-{opts.type}/")
    rdr: PathTracer = rdr_mapping[opts.type](emitter_configs, array_info, all_objs, configs)
    if type(rdr) != BDPT and configs.get('decomposition', 'none').startswith('transient'):
        CONSOLE.log("[bold yellow] Transient rendering is only supported in BDPT renderer.")

    max_iter_num = opts.iter_num if opts.iter_num > 0 else configs.get('iter_num', 2000)
    max_iter_num += 1                                   # we start from 1
    iter_cnt = 0
    
    eye_start = configs.get('start_t', 1)
    lit_start = configs.get('start_s', 0)
    max_bounce = configs.get('max_bounce', 16)
    max_depth = configs.get('max_depth', 16)
    eye_end = configs.get('end_t', max_bounce + 2)      # one more vertex for each path (starting vertex)
    lit_end = configs.get('end_s', max_bounce + 2)      # one more vertex for each path (starting vertex)
    CONSOLE.log(f"Starting to loop. Max eye :eyes: vnum: {eye_end}, max light :bulb: vnum: {lit_end}.")
    if opts.type == "bdpt":
        CONSOLE.log(f"BDPT with {max_bounce} bounce(s), max depth: {max_depth}. Cam vertices [{eye_start}, {eye_end}], light vertices [{lit_start}, {lit_end}]")
    else:
        CONSOLE.log(f"{name_mapping[opts.type]}Path Tracing with {max_bounce} bounce(s)")

    if opts.load:
        chkpt_path = os.path.join(folder_path(opts.chkpt_path), f"{opts.img_name}-{opts.name[:-4]}-{opts.type}.pkl")
        with open(chkpt_path, 'rb') as file:
            chkpt = pickle.load(file) 
        rdr.load_check_point(chkpt)

    CONSOLE.rule()
    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        SpinnerColumn(),
        BarColumn(),
        MofNCompleteColumn(),
        ItersPerSecColumn(suffix="fps"),
        TextColumn(" | ETA: "),
        TimeRemainingColumn(elapsed_when_finished=True),
        TextColumn(" | elasped: "),
        TimeElapsedColumn()
    )

    if opts.no_gui:
        try:
            with progress:
                for iter_cnt in progress.track(range(max_iter_num), description=""):
                    if opts.save_iter > 0 and (iter_cnt % opts.save_iter == 0):
                        chkpt = rdr.save_check_point()
                        save_check_point(chkpt, opts)
                    rdr.render(eye_start, eye_end, lit_start, lit_end, max_bounce, max_depth)
        except KeyboardInterrupt:
            if opts.save_iter > 0:
                chkpt = rdr.save_check_point()
                save_check_point(chkpt, opts)
            CONSOLE.log(":ok: Quit on Keyboard interruptions")
    else:
        window = tui.Window(f"{name_mapping[opts.type]}Path Tracing", res = (rdr.w, rdr.h))
        canvas = window.get_canvas()
        gui = window.get_gui()

        with progress:
            for iter_cnt in progress.track(range(1, max_iter_num), description=""):
                rdr.reset()
                for e in window.get_events(tui.PRESS):
                    if e.key == tui.ESCAPE:
                        window.running = False
                rdr.render(eye_start, eye_end, lit_start, lit_end, max_bounce, max_depth)
                if opts.save_iter > 0 and (iter_cnt % opts.save_iter == 0):
                    chkpt = rdr.get_check_point()
                    save_check_point(chkpt, opts)
                canvas.set_image(rdr.pixels)
                window.show()
                if window.running == False: 
                    if opts.save_iter > 0:
                        chkpt = rdr.get_check_point()
                        save_check_point(chkpt, opts)
                    break
                if output_freq > 0 and iter_cnt % output_freq == 0:
                    image = rdr.pixels.to_numpy()
                    ti.tools.imwrite(image, f"{output_folder}img_{iter_cnt:05d}.{opts.img_ext}")
    rdr.summary()
    if opts.profile:
        CONSOLE.rule()
        ti.profiler.print_kernel_profiler_info('trace') 
        CONSOLE.rule()
        ti.profiler.print_scoped_profiler_info()
        CONSOLE.rule()
        ti.profiler.memory_profiler.print_memory_profiler_info()
    image = apply_watermark(rdr, opts.normalize, True, not opts.no_watermark)
    save_figure = not opts.no_save_fig
    if save_figure:
        ti.tools.imwrite(image, f"{folder_path(opts.output_path)}{opts.img_name}-{opts.name[:-4]}-{opts.type}.{opts.img_ext}")
    if type(rdr) == BDPT and rdr.decomp[None] > 0:
        export_transient_profile(rdr, configs['sample_count'], opts.output_path, opts.name[:-4], opts.img_ext, opts.normalize, save_figure, opts.analyze)
