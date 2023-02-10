#-*-coding:utf-8-*-
"""
    Argument parser (from file and terminal)
    @author Qianyue He
    @date 2023.2.5
"""
import configargparse
from taichi import vulkan, cpu, cuda, gpu

__all__ = ['get_options', 'mapped_arch']

__MAPPING__ = {"vulkan": vulkan, "cpu": cpu, "cuda": cuda, "gpu": gpu}
mapped_arch = lambda x: __MAPPING__[x]

def get_options(delayed_parse = False):
    # IO parameters
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='Config file path')
    parser.add_argument("--iter_num",    default = -1, help = "Number of iterations (-1 means infinite)", type = int)
    parser.add_argument("--input_path",  default = "./inputs/", help = "Input scene file folder", type = str)
    parser.add_argument("--output_path", default = "./outputs/", help = "Output image file folder", type = str)
    parser.add_argument("--img_name",    default = "path-tracing.png", help = "Output image name", type = str)
    parser.add_argument("--scene",       default = "cbox", help = "Scene file name with extension", type = str)
    parser.add_argument("--name",       default = "complex.xml", help = "Scene file name with extension", type = str)
    parser.add_argument("--arch",        default = 'vulkan', choices=['cpu', 'gpu', 'vulkan', 'cuda'], help = "Backend-architecture")
    parser.add_argument("--profile",     default = False, action = "store_true", help = "Whether to profile the program")

    if delayed_parse:
        return parser
    return parser.parse_args()
