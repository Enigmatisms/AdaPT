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
    __type_mapping = {"det-refraction": 0, "null": -1, "lambertian": 1}
    def __init__(self, elem: xet.Element):
        super().__init__(elem, True)
        self.medium = Medium_np(elem.find("medium"))
        self.is_delta = False
        self.setup()
        if self.type_id == 0:
            self.is_delta = True
        
        # for BSDF, there will be medium defined in it

    def setup(self):
        if self.type not in BSDF_np.__type_mapping:
            raise NotImplementedError(f"Unknown BSDF type: {self.type}")
        self.type_id = BSDF_np.__type_mapping[self.type]

    def export(self):
        return BSDF(
            _type = self.type_id, is_delta = self.is_delta, k_d = vec3(self.k_d), 
            k_s = vec3(self.k_s), k_g = vec3(self.k_g), medium = self.medium.export()
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
    medium:     Medium          # attached medium

    # ========================= Deterministic Refraction =========================
    @ti.func
    def sample_det_refraction(self, it: ti.template(), incid: vec3, medium: ti.template(), mode):
        """ 
            Deterministic refraction sampling - Surface reflection is pure mirror specular \\ 
            other (Medium) could be incident medium or refraction medium, depending on \\
            whether the ray is entering or exiting the current BSDF
        """
        dot_normal = tm.dot(incid, it.n_s)
        # at least according to pbrt-v3, ni / nr is computed as the following (using shading normal)
        # see https://computergraphics.stackexchange.com/questions/13540/shading-normal-and-geometric-normal-for-refractive-surface-rendering
        entering_this = dot_normal < 0
        ni = ti.select(entering_this, medium.ior, self.medium.ior)
        nr = ti.select(entering_this, self.medium.ior, medium.ior)
        ret_pdf = 1.0
        ret_dir = vec3([0, 1, 0])
        ret_int = ti.select(it.is_tex_invalid(), self.k_d, it.tex)
        if is_total_reflection(dot_normal, ni, nr):
            ret_dir = (incid - 2 * it.n_s * dot_normal).normalized()
        else:
            refra_vec, cos_r2 = snell_refraction(incid, it.n_s, dot_normal, ni, nr)
            reflect_ratio = fresnel_equation(ni, nr, ti.abs(dot_normal), ti.sqrt(cos_r2))
            if ti.random(float) > reflect_ratio:        # refraction
                ret_pdf = 1. - reflect_ratio
                ret_dir = refra_vec
                if mode == TRANSPORT_RAD:       # non-symmetry effect
                    ret_int *= (ni * ni) / (nr * nr)
            else:                                       # reflection
                ret_dir = (incid - 2 * it.n_s * dot_normal).normalized()
                ret_pdf = reflect_ratio
        return ret_dir, ret_int * ret_pdf, ret_pdf
    
    @ti.func
    def eval_det_refraction(self, it: ti.template(), ray_in: vec3, ray_out: vec3, medium: ti.template(), mode):
        dot_out = tm.dot(ray_out, it.n_s)
        entering_this = dot_out < 0
        # notice that eval uses ray_out while sampling uses ray_in, therefore nr & ni have different order
        ni = ti.select(entering_this, medium.ior, self.medium.ior)
        nr = ti.select(entering_this, self.medium.ior, medium.ior)
        ret_int = vec3([0, 0, 0])
        diffuse_color = ti.select(it.is_tex_invalid(), self.k_d, it.tex)
        if is_total_reflection(dot_out, ni, nr):
            ref_dir = (ray_out - 2 * it.n_s * dot_out).normalized()
            if tm.dot(ref_dir, ray_in) > 1 - 5e-5:
                ret_int = diffuse_color
        else:
            # in sampling: ray_in points to the intersection, here ray_out points away from the intersection
            ref_dir = (ray_out - 2 * it.n_s * dot_out).normalized()
            refra_vec, cos_r2 = snell_refraction(ray_out, it.n_s, dot_out, ni, nr)
            if cos_r2 > 0.:
                reflect_ratio = fresnel_equation(ni, nr, ti.abs(dot_out), ti.sqrt(cos_r2))
                if tm.dot(refra_vec, ray_in) > 1 - 1e-4:            # ray_in close to refracted dir
                    ret_int = diffuse_color * (1. - reflect_ratio)
                    if mode == TRANSPORT_RAD:    # consider non-symmetric effect due to different transport mode
                        ret_int *= (ni * ni) / (nr * nr)
                elif tm.dot(ref_dir, ray_in) > 1 - 1e-4:            # ray_in close to reflected dir
                    ret_int = diffuse_color * reflect_ratio
            else:
                if tm.dot(ref_dir, ray_in) > 1 - 1e-4:            # ray_in close to reflected dir
                    ret_int = diffuse_color
        return ret_int
    
    # ========================= Lambertian transmission =======================
    @ti.func
    def sample_lambertian_trans(self, it: ti.template(), incid: vec3, medium: ti.template(), mode):
        """ 
            Deterministic refraction sampling - Surface reflection is pure mirror specular \\ 
            other (Medium) could be incident medium or refraction medium, depending on \\
            whether the ray is entering or exiting the current BSDF
        """
        dot_normal = tm.dot(incid, it.n_s)
        entering_this = dot_normal < 0
        ni = ti.select(entering_this, medium.ior, self.medium.ior)
        nr = ti.select(entering_this, self.medium.ior, medium.ior)
        ret_pdf = 1.0
        fresnel = 1.0
        is_delta = True
        ret_dir = vec3([0, 1, 0])
        ret_int = ti.select(it.is_tex_invalid(), self.k_d, it.tex)
        if is_total_reflection(dot_normal, ni, nr):
            ret_dir = (incid - 2 * it.n_s * dot_normal).normalized()
        else:
            ratio  = ni / nr
            cos_r2 = 1. - ti.pow(ratio, 2) * (1. - ti.pow(dot_normal, 2))        # refraction angle cosine
            reflect_ratio = fresnel_equation(ni, nr, ti.abs(dot_normal), ti.sqrt(cos_r2))
            if ti.random(float) > reflect_ratio:        # refraction
                fresnel = 1. - reflect_ratio
                local_new_dir, ret_pdf = cosine_hemisphere()
                ret_pdf *= fresnel
                normal = tm.sign(dot_normal) * it.n_s
                ret_dir, _R = delocalize_rotate(normal, local_new_dir)
                cosine_term = tm.max(0.0, tm.dot(normal, ret_dir))
                ret_int *= INV_PI * cosine_term
                if mode == TRANSPORT_RAD:       # non-symmetry effect
                    ret_int *= (ni * ni) / (nr * nr)
                is_delta = False
            else:                                       # reflection
                ret_dir = (incid - 2 * it.n_s * dot_normal).normalized()
                fresnel = reflect_ratio
                ret_pdf = reflect_ratio
        return ret_dir, ret_int * fresnel, ret_pdf, is_delta
    
    @ti.func
    def eval_lambertian_trans(self, it: ti.template(), ray_in: vec3, ray_out: vec3, medium: ti.template(), mode):
        dot_out = tm.dot(ray_out, it.n_s)
        entering_this = dot_out < 0
        # notice that eval uses ray_out while sampling uses ray_in, therefore nr & ni have different order
        ni = ti.select(entering_this, medium.ior, self.medium.ior)
        nr = ti.select(entering_this, self.medium.ior, medium.ior)
        ret_int = vec3([0, 0, 0])
        diffuse_color = ti.select(it.is_tex_invalid(), self.k_d, it.tex)
        if is_total_reflection(dot_out, ni, nr):
            ref_dir = (ray_out - 2 * it.n_s * dot_out).normalized()
            if tm.dot(ref_dir, ray_in) > 1 - 1e-4:
                ret_int = diffuse_color
        else:
            # in sampling: ray_in points to the intersection, here ray_out points away from the intersection
            ref_dir = (ray_out - 2 * it.n_s * dot_out).normalized()
            ratio      = ni / nr
            cos_r2     = 1. - ti.pow(ratio, 2) * (1. - ti.pow(dot_out, 2))        # refraction angle cosine
            dot_in = tm.dot(ray_in, it.n_s)
            if cos_r2 > 0.:
                reflect_ratio = fresnel_equation(ni, nr, ti.abs(dot_out), ti.sqrt(cos_r2))
                if dot_in * dot_out < 0:                            # reflection, ray_in and out same side
                    if tm.dot(ref_dir, ray_in) > 1 - 1e-4:            # ray_in close to reflected dir
                        ret_int = diffuse_color * reflect_ratio
                else:
                    ret_int = diffuse_color * ((1. - reflect_ratio) * INV_PI * ti.abs(dot_out))
                    if mode == TRANSPORT_RAD:    # consider non-symmetric effect due to different transport mode
                        ret_int *= (ni * ni) / (nr * nr)
            else:
                if tm.dot(ref_dir, ray_in) > 1 - 1e-4:            # ray_in close to reflected dir
                    ret_int = diffuse_color
        return ret_int
    # ========================= General operations =========================

    @ti.func
    def get_pdf(self, it: ti.template(), outdir: vec3, incid: vec3, medium: ti.template()):
        pdf = 0.
        if self._type == -1:
            pdf = ti.select(tm.dot(incid, outdir) > 1 - 1e-4, 1., 0.)
        else:
            dot_out = tm.dot(outdir, it.n_s)
            entering_this = dot_out < 0
            # notice that eval uses ray_out while sampling uses ray_in, therefore nr & ni have different order
            ni = ti.select(entering_this, medium.ior, self.medium.ior)
            nr = ti.select(entering_this, self.medium.ior, medium.ior)
            ref_dir = (outdir - 2 * it.n_s * dot_out).normalized()
            refra_vec, cos_r2 = snell_refraction(outdir, it.n_s, dot_out, ni, nr)
            if cos_r2 > 0.0:             # not total reflection, so there is not only one possible choice
                reflect_ratio = fresnel_equation(ni, nr, ti.abs(dot_out), ti.sqrt(cos_r2))
                if tm.dot(ref_dir, incid) > 1 - 1e-4:            # ray_in close to reflected dir
                    pdf = reflect_ratio
                else:
                    if self._type == 0 and tm.dot(refra_vec, incid) > 1 - 1e-4: # ray_in close to refracted dir
                        pdf = 1. - reflect_ratio
                    elif self._type == 1 and (tm.dot(incid, it.n_s) * dot_out > 0):
                        pdf = (1. - reflect_ratio) * ti.abs(dot_out) * INV_PI
            else:
                if tm.dot(ref_dir, incid) > 1 - 1e-4:
                    pdf = 1.
        return pdf
    
    @ti.func
    def is_non_null(self):          # null surface is -1
        return self._type >= 0
    
    # ========================= Surface interactions ============================
    @ti.func
    def eval_surf(self, it: ti.template(), incid: vec3, out: vec3, medium: ti.template(), mode) -> vec3:
        ret_spec = vec3([0, 0, 0])
        if self._type == 0:
            ret_spec = self.eval_det_refraction(it, incid, out, medium, mode)
        elif self._type == 1:
            ret_spec = self.eval_lambertian_trans(it, incid, out, medium, mode)
        return ret_spec
    
    @ti.func
    def sample_surf_rays(self, it: ti.template(), incid: vec3, medium: ti.template(), mode):
        ret_dir  = vec3([0, 0, 0])
        ret_spec = vec3([0, 0, 0])
        pdf      = 0.0
        is_delta = False
        if self._type == 0:
            ret_dir, ret_spec, pdf = self.sample_det_refraction(it, incid, medium, mode)
        elif self._type == 1:
            ret_dir, ret_spec, pdf, is_delta = self.sample_lambertian_trans(it, incid, medium, mode)
        return ret_dir, ret_spec, pdf, is_delta
    