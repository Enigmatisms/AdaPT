"""
    World definition
    @author: Qianyue He
    @date: 2023-2-7
"""

import sys
sys.path.append("..")

import numpy as np
import taichi as ti
import xml.etree.ElementTree as xet

from taichi.math import vec3
from bxdf.medium import Medium, Medium_np
from scene.general_parser import rgb_parse

class World_np:
    def __init__(self, elem: xet.Element):
        self.skybox = np.zeros(3, np.float32)
        self.ambient = np.zeros(3, np.float32)
        rgb_elems = elem.findall("rgb")
        for rgb_elem in rgb_elems:
            name = rgb_elem.get("name")
            if hasattr(self, name):
                self.__setattr__(name, rgb_parse(rgb_elem))
        self.medium = Medium_np(elem.find("medium"))
        self.C = 1.0
        print(f"World loading completed: \n{self}")

    def export(self):
        return World(skybox = vec3(self.skybox), ambient = vec3(self.ambient), medium = self.medium.export(), C = self.C)

    def __repr__(self):
        is_scattering = np.linalg.norm(self.medium.u_e) > 1e-4
        return f"<World with free space being [{self.medium.type_name.capitalize()}], ior: {self.medium.ior:.3f}, scatter: {is_scattering}>"

@ti.dataclass
class World:
    skybox:     vec3        # background color (texture is not supported yet)
    ambient:    vec3        # ambient light

    medium:     Medium      # medium in the free space (normally, transparent or vacuum)
    C:          ti.f32      # speed of light (could be 1. or 3e8)
