"""
    Bidirectional Scattering Distribution Function
    This is more general than BRDF, since it combines BRDF and BTDF
    the Idea of BSDF in here is more general: forward scattering and backward scattering of AN OBJECT
    but not just about a surface, therefore, each BSDF will have an attached medium
    the medium can either be scattering medium or non-scattering medium
    I... actually know nothing of this before...
    @author: Qianyue He
    @date: 2023-2-5
"""

import sys
sys.path.append("..")

import taichi as ti
import taichi.math as tm
import xml.etree.ElementTree as xet
from taichi.math import vec3

from la.geo_optics import *
from la.cam_transform import *
from sampler.general_sampling import *
from bxdf.brdf import BRDF_np
from bxdf.medium import Medium, Medium_np
from renderer.constants import TRANSPORT_RAD, INV_PI

__all__ = ['BSDF_np', 'BSDF']

class BSDF_np(BRDF_np):
    """
        BSDF base-class, 
        @author: Qianyue He
        @date: 2023-2-5
    """
    __type_mapping = {"det-refraction": 0, "null": -1}
    def __init__(self, elem: xet.Element):
        super().__init__(elem, True)
        self.medium = Medium_np(elem.find("medium"))
        self.is_delta = False
        self.setup()
        
        # for BSDF, there will be medium defined in it

    def setup(self):
        if self.type not in BSDF_np.__type_mapping:
            raise NotImplementedError(f"Unknown BSDF type: {self.type}")
        self.type_id = BSDF_np.__type_mapping[self.type]

    def export(self):
        return BSDF(
            _type = self.type_id, is_delta = self.is_delta, k_d = vec3(self.k_d), 
            k_s = vec3(self.k_s), k_g = vec3(self.k_g), k_a = vec3(self.k_a), medium = self.medium.export()
        )
        
    def __repr__(self) -> str:
        return f"<{self.type.capitalize()} BSDF with {self.medium.__repr__()} >"
    
# TODO: Non-symmetry Due to Refraction
@ti.dataclass
class BSDF:
    """
        TODO: 
        - implement simple BSDF first (simple refraction and mirror surface / glossy surface / lambertian surface)
        - transmission and reflection have independent distribution, yet transmission can be stochastic 
    """
    _type:      int
    is_delta:   int             # whether the BRDF is Dirac-delta-like
    k_d:        vec3            # diffusive coefficient (albedo)
    k_s:        vec3            # specular coefficient
    k_g:        vec3            # glossiness coefficient
    k_a:        vec3            # absorption coefficient
    medium:     Medium          # attached medium

    # ========================= Deterministic Refraction =========================
    @ti.func
    def sample_det_refraction(self, incid: vec3, normal: vec3, medium, mode):
        """ 
            Deterministic refraction sampling - Surface reflection is pure mirror specular \\ 
            other (Medium) could be incident medium or refraction medium, depending on \\
            whether the ray is entering or exiting the current BSDF
        """
        dot_normal = tm.dot(incid, normal)
        entering_this = dot_normal < 0
        ni = ti.select(entering_this, medium.ior, self.medium.ior)
        nr = ti.select(entering_this, self.medium.ior, medium.ior)
        ret_pdf = 1.0
        ret_dir = vec3([0, 1, 0])
        ret_int = self.k_d
        if is_total_reflection(dot_normal, ni, nr):
            # ret_dir = (incid - 2 * normal * dot_normal).normalized()      # we can account for total reflection... or not
            ret_int.fill(0.)
        else:
            refra_vec, _v = snell_refraction(incid, normal, dot_normal, ni, nr)
            reflect_ratio = frensel_equation(ni, nr, ti.abs(dot_normal), ti.abs(tm.dot(refra_vec, normal)))
            if ti.random(float) > reflect_ratio:        # refraction
                ret_pdf = 1. - reflect_ratio
                ret_dir = refra_vec
                if mode == TRANSPORT_RAD:       # non-symmetry effect
                    ret_int *= (ni * ni) / (nr * nr)
            else:                                       # reflection
                ret_dir = (incid - 2 * normal * dot_normal).normalized()
                ret_pdf = reflect_ratio
        return ret_dir, ret_int * ret_pdf, ret_pdf
    
    @ti.func
    def eval_det_refraction(self, ray_in: vec3, ray_out: vec3, normal: vec3, medium, mode):
        dot_out = tm.dot(ray_out, normal)
        entering_this = dot_out < 0
        # notice that eval uses ray_out while sampling uses ray_in, therefore nr & ni have different order
        ni = ti.select(entering_this, medium.ior, self.medium.ior)
        nr = ti.select(entering_this, self.medium.ior, medium.ior)
        ret_int = vec3([0, 0, 0])
        if is_total_reflection(dot_out, ni, nr):
            ref_dir = (ray_out - 2 * normal * dot_out).normalized()
            if tm.dot(ref_dir, ray_in) > 1 - 5e-5:
                ret_int = self.k_d
        else:
            # in sampling: ray_in points to the intersection, here ray_out points away from the intersection
            ref_dir = (ray_out - 2 * normal * dot_out).normalized()
            refra_vec, valid_ref = snell_refraction(ray_out, normal, dot_out, ni, nr)
            if valid_ref:
                reflect_ratio = frensel_equation(ni, nr, ti.abs(dot_out), ti.abs(tm.dot(refra_vec, normal)))
                if tm.dot(refra_vec, ray_in) > 1 - 5e-5:            # ray_in close to refracted dir
                    ret_int = self.k_d * (1. - reflect_ratio)
                    if mode == TRANSPORT_RAD:    # consider non-symmetric effect due to different transport mode
                        ret_int *= (ni * ni) / (nr * nr)
                elif tm.dot(ref_dir, ray_in) > 1 - 5e-5:            # ray_in close to reflected dir
                    ret_int = self.k_d * reflect_ratio
            else:
                if tm.dot(ref_dir, ray_in) > 1 - 5e-5:            # ray_in close to reflected dir
                    ret_int = self.k_d
        return ret_int
    # ========================= General operations =========================

    @ti.func
    def get_pdf(self, outdir: vec3, normal: vec3, incid: vec3, medium):
        pdf = 0.
        if self._type == -1:
            pdf = ti.select(tm.dot(incid, outdir) > 1 - 5e-5, 1., 0.)
        elif self._type == 0:
            dot_out = tm.dot(outdir, normal)
            entering_this = dot_out < 0
            # notice that eval uses ray_out while sampling uses ray_in, therefore nr & ni have different order
            ni = ti.select(entering_this, medium.ior, self.medium.ior)
            nr = ti.select(entering_this, self.medium.ior, medium.ior)
            ref_dir = (outdir - 2 * normal * dot_out).normalized()
            refra_vec, valid_refra = snell_refraction(outdir, normal, dot_out, ni, nr)
            if valid_refra:             # not total reflection, so there is not only one possible choice
                reflect_ratio = frensel_equation(ni, nr, ti.abs(dot_out), ti.abs(tm.dot(refra_vec, normal)))
                if tm.dot(refra_vec, incid) > 1 - 5e-5:            # ray_in close to refracted dir
                    pdf = 1. - reflect_ratio
                elif tm.dot(ref_dir, incid) > 1 - 5e-5:            # ray_in close to reflected dir
                    pdf = reflect_ratio
            else:
                if tm.dot(ref_dir, incid) > 1 - 5e-5:
                    pdf = 1.
        return pdf
    
    @ti.func
    def is_non_null(self):          # null surface is -1
        return self._type >= 0
    
    # ========================= Surface interactions ============================
    @ti.func
    def eval_surf(self, incid: vec3, out: vec3, normal: vec3, medium, mode) -> vec3:
        ret_spec = vec3([0, 0, 0])
        if self._type == 0:
            ret_spec = self.eval_det_refraction(incid, out, normal, medium, mode)
        return ret_spec
    
    @ti.func
    def sample_surf_rays(self, incid: vec3, normal: vec3, medium, mode):
        # TODO: we need mode here (and in eval)
        ret_dir  = vec3([0, 0, 0])
        ret_spec = vec3([0, 0, 0])
        pdf      = 0.0
        if self._type == 0:
            ret_dir, ret_spec, pdf = self.sample_det_refraction(incid, normal, medium, mode)
        return ret_dir, ret_spec, pdf
    