#-*-coding:utf-8-*-
"""
    Argument parser (from file and terminal)
    @author Qianyue He
    @date 2023.2.5
"""
import configargparse
from taichi import vulkan, cpu, cuda, gpu

__all__ = ['get_options', 'mapped_arch', 'get_tdom_options']

__MAPPING__ = {"vulkan": vulkan, "cpu": cpu, "cuda": cuda, "gpu": gpu}
mapped_arch = lambda x: __MAPPING__[x]

def get_options(delayed_parse = False):
    # IO parameters
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='Config file path')
    parser.add_argument("--iter_num",      default = 512, help = "Number of iterations (-1 means infinite)", type = int)
    parser.add_argument("--normalize",     default = 0., help = "Normalize the output picture with its <x> quantile value", type = float)
    parser.add_argument("--output_freq",   default = 0, help = "Whether to output intermediate results (0 means no)", type = int)
    parser.add_argument("--input_path",    default = "./inputs/", help = "Input scene file folder", type = str)
    parser.add_argument("--output_path",   default = "./outputs/", help = "Output image file folder", type = str)
    parser.add_argument("--img_name",      default = "pbr", help = "Output image name", type = str)
    parser.add_argument("--img_ext",       default = "png", choices=['png', 'jpg', 'bmp'], help = "Output image extension", type = str)
    parser.add_argument("--scene",         default = "cbox", help = "Name of the scene", type = str)
    parser.add_argument("--name",          default = "complex.xml", help = "Scene file name with extension", type = str)
    parser.add_argument("--arch",          default = 'cuda', choices=['cpu', 'gpu', 'vulkan', 'cuda'], help = "Backend-architecture")
    parser.add_argument("--type",          default = 'vpt', choices=['vpt', 'pt', 'bdpt'], help = "Algorithm to be used")
    
    parser.add_argument("-p", "--profile", default = False, action = "store_true", help = "Whether to profile the program")
    parser.add_argument("--no_gui",        default = False, action = "store_true", help = "Whether to display GUI")
    parser.add_argument("-d", "--debug",   default = False, action = "store_true", help = "Whether to debug taichi kernel")
    parser.add_argument("-a", "--analyze", default = False, action = "store_true", help = "Whether to analyze transient rendering time domain")
    parser.add_argument("--no_cache",      default = False, action = "store_true", help = "Whether to cache JIT compilation")
    parser.add_argument("--no_save_fig",   default = False, action = "store_true", help = "Whether to save images")
    parser.add_argument("--no_watermark",  default = False, action = "store_true", help = "Whether to add watermark")

    if delayed_parse:
        return parser
    return parser.parse_args()

def get_tdom_options(delayed_parse = False):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file = True, help='Config file path')
    parser.add_argument("--sim_path",      default = "./analysis/", help = "Input simulation data folder", type = str)
    parser.add_argument("--theory_path",   default = "./other_data/", help = "Input theoretical data folder", type = str)
    parser.add_argument("--real_path",     default = "./other_data/", help = "Input realistic data folder", type = str)
    parser.add_argument("--sim_name",      default = "foam3-200-diag_tri.data", help = "Input simulation data name", type = str)
    parser.add_argument("--theory_name",   default = "diffuse_3cm.mat", help = "Input theoretical data name", type = str)
    parser.add_argument("--real_name",     default = "transient_3cm.mat", help = "Input SPAD transient data name", type = str)
    parser.add_argument("--output_folder", default = "./output/", help = "Output folder of time analysis results", type = str)

    parser.add_argument("--mode",          default = "sim", choices=['sim', 'mat', 'comp'], help = "Evaluation mode", type = str)
    parser.add_argument("--window_mode",   default = "whole", choices=['diag_tri', 'diag_side_mean', 'whole'], help = "Window cropping mode", type = str)
    parser.add_argument("--sim_interval",  default = 0.001, help = "Time interval of simulated samples", type = float)
    parser.add_argument("--sim_samples",   default = 400, help = "Number of simulated samples", type = float)
    parser.add_argument("--sim_sol",       default = 1.0, help = "Speed of light for simulation", type = float)

    parser.add_argument("--prominence",    default = 0.2, help = "Peak finding parameter", type = float)

    parser.add_argument("--analyze_peak",  default = False, action = "store_true", help = "Whether to analyze peak data")
    parser.add_argument("--save_fig",      default = False, action = "store_true", help = "Whether to save figure instead of displaying")
    parser.add_argument("--show_real",     default = False, action = "store_true", help = "Diffusion & Real data, which to show")

    if delayed_parse:
        return parser
    return parser.parse_args()
