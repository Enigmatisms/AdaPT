"""
    Grid Volume Scattering Medium (RGB / Float)
    @author: Qianyue He
    @date: 2024-6-22
"""

import numpy as np
import taichi as ti
import xml.etree.ElementTree as xet

from taichi.math import vec3

from bxdf.phase import PhaseFunction
from bxdf.medium import Medium_np
from la.cam_transform import delocalize_rotate
from parsers.general_parser import get, rgb_parse
from sampler.general_sampling import random_rgb
from rich.console import Console
CONSOLE = Console(width = 128)

try:
    from vol_loader import vol_file_to_numpy
except ImportError:
    CONSOLE.log("[red]:skeleton: Error [/red]: vol_loader is not found, possible reason:")
    CONSOLE.log(f"module not installed. Use 'python ./setup.py install --user' in {'./bxdf/'}")
    raise ImportError("vol_loader not found")

""" TODO: add transform parsing
"""
class GridVolume_np:
    __type_mapping = {"none": 0, "mono": 1, "rgb": 2}
    def __init__(self, elem: xet.Element, is_world = False):
        self.albedo = np.ones(3, np.float32)
        self.par = np.zeros(3, np.float32)
        self.pdf = np.float32([1., 0., 0.])
        self.phase_type_id = -1
        self.phase_type = "hg"
        self.type_id = -1
        self.type_name = "none"

        self.xres = 0
        self.yres = 0
        self.zres = 0
        self.channel = 0
        self.grids = None

        elem_to_query = {"rgb": rgb_parse, "float": lambda el: get(el, "value"), "string": lambda path: vol_file_to_numpy(path)}
        if elem is not None:
            type_name = elem.get("type")
            if type_name in GridVolume_np.__type_mapping:
                self.type_id = GridVolume_np.__type_mapping[type_name]
            else:
                raise NotImplementedError(f"GridVolume type '{type_name}' is not supported.")
            self.type_name = type_name 

            phase_type = elem.get("phase_type")
            if type_name in Medium_np.__type_mapping:
                self.type_id = Medium_np.__type_mapping[phase_type]
            else:
                raise NotImplementedError(f"Phase function type '{phase_type}' is not supported.")
            
            for tag, query_func in elem_to_query.items():
                tag_elems = elem.findall(tag)
                for tag_elem in tag_elems:
                    name = tag_elem.get("name")
                    if hasattr(self, name):
                        self.__setattr__(name, query_func(tag_elem))
        else:
            if not is_world:
                CONSOLE.log("[yellow]:warning: Warning: default initialization yields <transparent>, which is a trivial medium.")
        self.u_e = self.u_a + self.u_s

    def setup_volume(self, path:str):
        self.grids, (self.xres, self.yres, self.zres, self.channel) = vol_file_to_numpy(path)
        CONSOLE.log(f"Volume grid loaded, type {self.type_name},\n \
                    shape: (x = {self.xres}, y = {self.yres}, z = {self.zres}, c = {self.channel})")
    
    def export(self):
        pass
    
    def __repr__(self):
        return f"<Volume grid {self.type_name.upper()} with phase {self.phase_type}, albedo: {self.albedo},\n \
                shape: (x = {self.xres}, y = {self.yres}, z = {self.zres}, c = {self.channel})>"

@ti.dataclass
class GridVolume:
    _type:  int
    albedo: vec3            # scattering
    