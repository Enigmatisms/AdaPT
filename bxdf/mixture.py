"""
    BxDF mixture struct 
    @author: Qianyue He
    @date: 2023.7.18

    NOTE: mixture model (BxDF mixture) is only available in branch 'more'
    since mixture model will add compile-time and run-time overhead
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

__PDF_EPS__ = 1e-5

@ti.dataclass
class BxDFMixture:
    """ This struct can be used to construct coating
        and plastic BSDF
    """
    diffuse: int
    """ BxDF index for diffusive part """
    glossy: int
    """ BxDF index for glossy part"""
    specular: int
    """ BxDF index for specular part"""
    transmit: int
    """ BTDF index for transmission part"""

    p_d: float
    """ PDF for sampling diffusive part """
    p_g: float
    """ PDF for sampling glossy part """
    p_s: float
    """ PDF for sampling specular part """
    p_t: float
    """ PDF for sampling transmission part """
    acc_proba: vec2
    """ accumulated proba for faster branching: (p_d + p_g, p_d + p_g + p_s) """

    @ti.func
    def sample(
        self, brdf: ti.template(), bsdf: ti.template(), 
        it:ti.template(), incid: vec3, medium: ti.template(), mode: int
    ):
        """ select one component to sample from """
        component_eps = ti.random(float)
        proba = 1.0
        component_idx = 0
        reflect_only = True
        if component_eps >= self.acc_proba[0]:
            if component_eps < self.acc_proba[1]:
                proba = self.p_s
                component_idx = 2
            else:
                proba = self.p_t
                component_idx = 3
                reflect_only = False
        else:
            if component_eps < self.p_d:
                proba = self.p_d
                component_idx = 0
            else:
                proba = self.p_g
                component_idx = 1

        ret_dir  = vec3([0, 1, 0])
        ret_spec = vec3([1, 1, 1])
        pdf      = 1.0
        if reflect_only:
            ret_dir, ret_spec, pdf = brdf[component_idx].sample_new_rays(it, incid)
        else:
            ret_dir, ret_spec, pdf = bsdf[component_idx].sample_surf_rays(it, incid, medium, mode)
        pdf *= proba
        return ret_dir, ret_spec, pdf
    
    @ti.func
    def eval(
        self, brdf: ti.template(), bsdf: ti.template(), it:ti.template(), 
        incid: vec3, out: vec3, medium: ti.template(), mode: int
    ):
        """ Evaluate contribution according all of the components """
        pdfs = vec4([self.p_d, self.p_g, self.p_s])
        ret_spec = ZERO_V3
        for i in range(3):
            pdf = pdfs[i]
            if pdf <= __PDF_EPS__: continue
            ret_spec += brdf[i].eval(it, incid, out) * pdf
        if self.p_t > __PDF_EPS__:
            ret_spec += bsdf[i].eval_surf(it, incid, out, medium, mode) * self.p_t
        return ret_spec

    @ti.func
    def get_pdf(
        self, brdf: ti.template(), bsdf: ti.template(), 
        it:ti.template(), incid: vec3, out: vec3, medium: ti.template()
    ):
        """ Evaluate PDF according all of the components """
        pdfs = vec4([self.p_d, self.p_g, self.p_s])
        ret_pdf = 0
        for i in range(3):
            pdf = pdfs[i]
            if pdf <= __PDF_EPS__: continue
            ret_pdf += brdf[i].get_pdf(it, out, incid) * pdf
        if self.p_t > __PDF_EPS__:
            ret_pdf += bsdf[i].get_pdf(it, out, incid, medium) * self.p_t
        return ret_pdf
