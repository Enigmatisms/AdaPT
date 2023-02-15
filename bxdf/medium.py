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
from bxdf.phase import PhaseFunction
from scene.general_parser import get, rgb_parse
from sampler.general_sampling import random_rgb

__all__ = ['Medium', 'Medium_np']

class Medium_np:
    __type_mapping = {"hg": 0, "multi-hg": 1, "rayleigh": 2, "mie": 3, "transparent": -1}
    def __init__(self, elem: xet.Element, is_world = False):
        """
            Without spectral information, Rayleigh scattering here might not be physically-based
        """
        self.ior = 1.0
        self.u_a = np.zeros(3, np.float32)
        self.u_s = np.zeros(3, np.float32)
        self.par = np.zeros(3, np.float32)
        self.pdf = np.float32([1., 0., 0.])
        self.type_id = -1
        self.type_name = "transparent"

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
                print("Warning: default initialization yields <transparent>, which is a trivial medium.")
        self.u_e = self.u_a + self.u_s
    
    def export(self):
        phase_func = PhaseFunction(_type = self.type_id, par = vec3(self.par), pdf = vec3(self.pdf))
        return Medium(_type = self.type_id, ior = self.ior, u_a = vec3(self.u_a), 
            u_s = vec3(self.u_s), u_e = vec3(self.u_e), ph = phase_func
        )
    
    def __repr__(self):
        return f"<Medium {self.type_name.capitalize()} with ior {self.ior:.3f}, extinction: {self.u_e}>"

@ti.dataclass
class Medium:
    _type:  ti.i32
    ior:    ti.f32
    u_s:    vec3            # scattering
    u_a:    vec3            # absorption
    u_e:    vec3            # precomputed extinction
    ph:     PhaseFunction   # phase function

    @ti.func
    def is_scattering(self):   # check whether the current medium is scattering medium
        return self._type >= 0
    
    @ti.func
    def transmittance(self, depth: ti.f32):
        is_scattering = self._type >= 0
        transmittance = ti.exp(-self.u_e * depth)
        # transmitted without being scattered (PDF)
        transmittance /= self.pdf_no_scatter(depth)
        return is_scattering, transmittance
    
    @ti.func
    def pdf_no_scatter(self, depth):
        return ti.max(ti.exp(-self.u_e * depth).sum() / 3., 1e-5)      # self.u_e * depth < 11.5 (log 1e-5 \approx -11.5)

    @ti.func
    def sample_mfp(self, max_depth):
        random_ue = random_rgb(self.u_e)
        sample_t = - ti.log(1. - ti.random(ti.f32)) / random_ue
        beta = vec3([1., 1., 1.])
        is_medium_interact = False
        if sample_t >= max_depth:
            sample_t = max_depth
            pdf = self.pdf_no_scatter(max_depth)
            beta =  ti.exp(-self.u_e * max_depth) / pdf
        else:
            is_medium_interact = True
            pdf = (self.u_e * ti.exp(-self.u_e * sample_t)).sum() / 3.
            beta =  ti.exp(-self.u_e * sample_t) * self.u_s / pdf
        return is_medium_interact, sample_t, beta