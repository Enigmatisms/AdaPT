"""
    BxDF mixture struct 
    @author: Qianyue He
    @date: 2023.7.18

    Mixtures and objects are bi-jection:
    we create one mixture for each object
    TODO: the parsing code should be heavily modified
"""

import sys
sys.path.append("..")

import numpy as np
import taichi as ti
import taichi.math as tm
import xml.etree.ElementTree as xet
from taichi.math import vec2, vec3, vec4

from la.geo_optics import *
from la.cam_transform import *
from sampler.general_sampling import *
from parsers.general_parser import rgb_parse
from renderer.constants import INV_PI, ZERO_V3, INVALID

from rich.console import Console
CONSOLE = Console(width = 128)

""" The best practice for this is to use mixture-based BSDF organization
    for every mixture (in xml) we create a Taichi mixture struct
    for every isolated BRDF (with no mixture reference) we create a single mixture
    and use diffuse to store the component
"""

vec4i = ti.types.vector(3, int)

__PDF_EPS__ = 1e-5

class BxDFMixture_np:
    """ This is python end mixture definition
    """
    def __init__(self, elem: xet.Element) -> None:
        pass

@ti.dataclass
class BxDFMixture:
    """ This struct can be used to construct coating
        and plastic BSDF
    """
    comps: vec4i
    """ Component idx (mapping), diffusive, glossy, specular, transmission 
        Valid when non-negative
    """
    proba: vec4
    """ Component sampling PDF: p_d, p_g, p_s, p_t """

    acc_proba: vec2
    """ Precomputed accumulated proba for faster branching: (p_d + p_g, p_d + p_g + p_s) """

    @ti.func
    def sample(self):
        """ select one component to sample from
            returned value can be directly used in path_tracer without
            calling some functions twice (incuring compile-time overhead)
        """
        component_eps = ti.random(float)
        proba = 1.0
        component_idx = -1
        reflect_only = True
        if component_eps >= self.acc_proba[0]:
            if component_eps < self.acc_proba[1]:
                proba = self.proba[2]
                component_idx = self.comps[2]
            else:
                proba = self.proba[3]
                component_idx = self.comps[3]
                reflect_only = False
        else:
            if component_eps < self.proba[0]:
                proba = self.proba[0]
                component_idx = self.comps[0]
            else:
                proba = self.proba[1]
                component_idx = self.comps[1]
        return proba, component_idx, reflect_only
    
    @ti.func
    def eval(
        self, brdf: ti.template(), bsdf: ti.template(), it:ti.template(), 
        incid: vec3, out: vec3, medium: ti.template(), mode: int, two_sides: bool = True,
    ):
        """ Evaluate contribution according all of the components """
        ret_spec = ZERO_V3
        dot_res = 1.0
        if two_sides:
            dot_res = tm.dot(incid, it.n_s)
            if dot_res > 0.:                    # two sides
                it.n_s *= -1
                it.n_g *= -1
        for i in range(3):
            pdf = self.proba[i]
            if pdf <= __PDF_EPS__: continue
            ret_spec += brdf[i].eval(it, incid, out) * pdf
        if self.proba[3] > __PDF_EPS__:
            if two_sides:
                if dot_res > 0.:                # two sides - convert back to original direction
                    it.n_s *= -1
                    it.n_g *= -1
            ret_spec += bsdf[i].eval_surf(it, incid, out, medium, mode) * self.proba[3]
        return ret_spec

    @ti.func
    def get_pdf(
        self, brdf: ti.template(), bsdf: ti.template(), it:ti.template(), 
        incid: vec3, out: vec3, medium: ti.template(), two_sides: bool = True,
    ):
        """ Evaluate PDF according all of the components """
        ret_pdf = 0
        dot_res = 1.0
        if two_sides:
            dot_res = tm.dot(incid, it.n_s)
            if dot_res > 0.:                    # two sides
                it.n_s *= -1
                it.n_g *= -1
        for i in range(3):
            pdf = self.proba[i]
            if pdf <= __PDF_EPS__: continue
            ret_pdf += brdf[i].get_pdf(it, out, incid) * pdf
        if self.proba[3] > __PDF_EPS__:
            if two_sides:
                if dot_res > 0.:                # two sides - convert back to original direction
                    it.n_s *= -1
                    it.n_g *= -1
            ret_pdf += bsdf[i].get_pdf(it, out, incid, medium) * self.proba[3]
        return ret_pdf
