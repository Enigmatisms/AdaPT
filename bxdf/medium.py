"""
    Volume Scattering Medium
    For one given object, object can have [BSDF + Medium], [BSDF], [BRDF], which corresponds to
    (1) non-opaque object with internal scattering (2) non-opaque object without scattering (3) opaque object
    @author: Qianyue He
    @date: 2023-2-7
"""

import numpy as np
import taichi as ti
import xml.etree.ElementTree as xet

from taichi.math import vec3
from scene.general_parser import get, rgb_parse

__all__ = ['Medium', 'Medium_np']

class Medium_np:
    __type_mapping = {"transparent": 0, "h-g": 1, "rayleigh": 2, "mie": 3, "air": -1}
    def __init__(self, elem: xet.Element, is_world = False):
        self.ior = 1.0
        self.u_a = np.zeros(3, np.float32)
        self.u_s = np.zeros(3, np.float32)
        self.par = np.zeros(3, np.float32)
        self.type_id = -1
        self.type_name = "air"

        elem_to_query = {"rgb": rgb_parse, "float": lambda el: get(el, "value")}
        if elem is not None:
            type_name = elem.get("type")
            if type_name in Medium_np.__type_mapping:
                self.type_id = Medium_np.__type_mapping[type_name]
            else:
                raise NotImplementedError(f"Medium type '{type_name}' is not supported.")
            self.type_name = type_name
            for tag, query_func in elem_to_query.items():
                tag_elems = elem.findall(tag)
                for tag_elem in tag_elems:
                    name = tag_elem.get("name")
                    if hasattr(self, name):
                        self.__setattr__(name, query_func(tag_elem))
        else:
            if not is_world:
                print("Warning: default initialization yields air, which is a trivial medium.")
        self.u_e = self.u_a + self.u_s
    
    def export(self):
        return Medium(_type = self.type_id, ior = self.ior, u_a = vec3(self.u_a), 
            u_s = vec3(self.u_s), u_e = vec3(self.u_e), params = vec3(self.par)
        )
    
    def __repr__(self):
        return f"<Medium {self.type_name.capitalize()} with ior {self.ior:.3f}, extinction: {self.u_e}>"

@ti.dataclass
class Medium:
    _type:  ti.i32
    ior:    ti.f32
    u_s:    vec3      # scattering
    u_a:    vec3      # absorption
    u_e:    vec3      # precomputed extinction
    params: vec3      # other parameters (like phase function)

    """ All the functions related to 'directions' are useless unless the medium is a scattering one """
    def sample_direction(self):
        pass

    def eval_direction(self):
        pass

    def pdf_direction(self):
        pass
    