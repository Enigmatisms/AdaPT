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
from parsers.general_parser import get
from renderer.constants import INV_PI, ZERO_V3, INVALID

from rich.console import Console
CONSOLE = Console(width = 128)

""" The best practice for this is to use mixture-based BSDF organization
    for every mixture (in xml) we create a Taichi mixture struct
    for every isolated BRDF (with no mixture reference) we create a single mixture
    and use diffuse to store the component
"""

vec4i = ti.types.vector(4, int)

__PDF_EPS__ = 1e-5

class BxDFMixture_np:
    """ This is python end mixture definition
    """
    __order_map__ = {"diffuse": 0, "glossy": 1, "specular": 2, "transmission": 3}
    def __init__(self, elem: xet.Element = None, ref_dict: dict = None) -> None:
        self.id  = "invalid"
        self.single_comp = -1
        self.ref_ids   = []
        self.pdfs      = np.zeros(4, dtype = np.float32)
        self.cdfs      = np.zeros(4, dtype = np.float32)
        self.comps     = np.full(4, -1, dtype = np.int32)
        if elem is not None and ref_dict is not None:
            self.setup(elem, ref_dict)

    @staticmethod
    def from_single(name: str, bxdf_id: int, is_brdf: bool = True):
        mixture = BxDFMixture_np()
        mixture.id = name
        if is_brdf:
            mixture.single_comp = 0 
            mixture.pdfs  = np.float32([1, 0, 0, 0])
            mixture.comps = np.int32([bxdf_id, -1, -1, -1])
        else:
            mixture.single_comp = 3 
            mixture.pdfs = np.float32([0, 0, 0, 1])
            mixture.comps = np.int32([-1, -1, -1, bxdf_id])
        return mixture

    def setup(self, elem: xet.Element, ref_dict: dict):
        self.id  = elem.get("id")
        all_refs = elem.findall("ref") 
        for ref in all_refs:
            ty = ref.get("type")
            if ty not in BxDFMixture_np.__order_map__:
                CONSOLE.log(f"[yellow]Warning: [/yellow]Mixture <{self.id}> has unsupported type '{ty}'. Fall back to 'diffuse'.")
                ty = "diffuse"
            index = BxDFMixture_np.__order_map__[ty]
            ref_pdf = get(ref, "pdf")
            # TODO: we need a reference dict

            ref_id = ref.get("id")
            if ref_id not in ref_dict:
                CONSOLE.log(f"[bold red]:skull: Key error exception raised during processing mixture"
                            "'{self.id}'. ref_id <{ref_id}> not found in all BSDFs.")
                exit(1)
            self.comps[index] = ref_dict[ref_id]
            self.ref_ids.append(ref_id)
            if ref_pdf < 1e-5:
                CONSOLE.log(f"[yellow]Warning: [/yellow]Mixture <{self.id}> has a component almost impossible to sample: '{ty}'.")
                ref_pdf = 0
            self.pdfs[index]  = ref_pdf
        self.cdfs = np.cumsum(self.pdfs)

    def export(self):
        """ Export to taichi end """
        return BxDFMixture(comps = vec4i(self.comps), proba = vec4(self.pdfs), 
                           acc_proba = vec2(self.cdfs[1:3]), single_comp = self.single_comp
        )
    
    def __repr__(self) -> str:
        return f"<{self.id} Mixture, single component: {self.single_comp}, references: {self.ref_ids}, comps: {self.comps}>"

@ti.dataclass
class BxDFMixture:
    """ This struct can be used to construct coating 
        and plastic BSDF
    """
    comps: vec4i
    """ Component idx (mapping), diffusive, glossy, specular, transmission, 
        valid when non-negative
    """
    proba: vec4
    """ Component sampling PDF: p_d, p_g, p_s, p_t. We can actually always opt for uniform distribution \\
        However, there are indeed easy-to-sample / hard-to-sample components
    """ 

    acc_proba: vec2
    """ Precomputed accumulated proba for faster branching: (p_d + p_g, p_d + p_g + p_s) """

    single_comp: int
    """ if there is only one component, single_comp will be non-negative """


    @ti.func
    def sample(self):
        """ select one component to sample from 
            returned value can be directly used in path_tracer without 
            calling some functions twice (incuring compile-time overhead)
        """
        proba = 1.0
        component_idx = -1
        if self.single_comp >= 0:
            # actually, single component can be BRDF (diffuse) or BSDF (transmission)
            # for a correct MIS computation
            component_idx = self.comps[self.single_comp]
        else:
            component_eps = ti.random(float)
            if component_eps >= self.acc_proba[0]:
                if component_eps < self.acc_proba[1]:
                    proba = self.proba[2]
                    component_idx = self.comps[2]
                else:
                    proba = self.proba[3]
                    component_idx = self.comps[3]
            else:
                if component_eps < self.proba[0]:
                    proba = self.proba[0]
                    component_idx = self.comps[0]
                else:
                    proba = self.proba[1]
                    component_idx = self.comps[1]
        return proba, component_idx
    
    @ti.func
    def eval(
        self, brdf: ti.template(), bsdf: ti.template(), it:ti.template(), 
        incid: vec3, out: vec3, medium: ti.template(), mode: int, two_sides: bool = True,
    ):
        """ Evaluate contribution according all of the components """
        ret_spec = ZERO_V3
        dot_res = 1.0
        index = self.comps[3]
        if index >= 0:
            ret_spec += bsdf[index].eval_surf(it, incid, out, medium, mode)

        if two_sides:
            dot_res = tm.dot(incid, it.n_s)
            if dot_res > 0.:                    # two sides
                it.n_s *= -1
                it.n_g *= -1
        for i in range(3):
            index = self.comps[i]
            if index < 0: continue
            ret_spec += brdf[index].eval(it, incid, out)
        return ret_spec

    @ti.func
    def get_pdf(
        self, brdf: ti.template(), bsdf: ti.template(), it:ti.template(), 
        incid: vec3, out: vec3, medium: ti.template(), two_sides: bool = True,
    ):
        """ Evaluate PDF according all of the components """
        ret_pdf = 0.
        dot_res = 1.0
        index = self.comps[3]
        if index >= 0:
            ret_pdf += bsdf[index].get_pdf(it, out, incid, medium) * self.proba[3]
        if two_sides:
            dot_res = tm.dot(incid, it.n_s)
            if dot_res > 0.:                    # two sides
                it.n_s *= -1
                it.n_g *= -1
        for i in range(3):
            index = self.comps[i]
            if index < 0: continue
            ret_pdf += brdf[index].get_pdf(it, out, incid) * self.proba[i]
        return ret_pdf
    
    @ti.func
    def is_delta(self, brdf: ti.template(), bsdf: ti.template()):
        """ if all the component is delta then the BxDF mixture is delta """
        is_delta = True
        index = self.comps[3]
        if index >= 0:
            is_delta &= bsdf[index].is_delta
        if is_delta:
            for i in range(3):
                index = self.comps[i]
                if index < 0: continue
                is_delta &= brdf[index].is_delta
                if not is_delta: break
        return is_delta
    
    @ti.func
    def non_null_index(self):
        """ if transmittance is null and there is no other components, then it is null surface """
        return ti.select(self.single_comp == 3, self.comps[3], -1)
