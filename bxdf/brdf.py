"""
    All the BRDFs are here, note that only three kinds of simple BRDF are supported
    Blinn-Phong / Lambertian / Mirror specular / Modified Phong / Frensel Blend
    @author: Qianyue He
    @date: 2023-1-23
"""
import sys
sys.path.append("..")

import numpy as np
import taichi as ti
import taichi.math as tm
import xml.etree.ElementTree as xet
from taichi.math import vec3, vec4, mat3

from la.geo_optics import *
from la.cam_transform import *
from sampler.general_sampling import *
from scene.general_parser import rgb_parse

__all__ = ['BRDF_np', 'BRDF']

EPS = 1e-7
INV_PI = 1. / tm.pi

class BRDF_np:
    """
        BRDF base-class, 
        @author: Qianyue He
        @date: 2023-1-23
    """
    __all_albedo_name       = {"reflectance", "albedo", "k_d"}
    __all_glossiness_name   = {"glossiness", "shininess", "k_g"}
    __all_specular_name     = {"specular", "k_s"}
    __all_absorption_name   = {"absorptions", "k_a"}
    # Attention: microfacet support will not be added recently
    __type_mapping          = {"blinn-phong": 0, "lambertian": 1, "specular": 2, "microfacet": 3, "mod-phong": 4, "frensel-blend": 5}
    
    def __init__(self, elem: xet.Element, no_setup = False):
        self.type: str = elem.get("type")
        self.type_id = -1
        self.id: str = elem.get("id")
        self.k_d = np.ones(3, np.float32)
        self.k_s = np.zeros(3, np.float32)
        self.k_g = np.ones(3, np.float32)
        self.k_a = np.zeros(3, np.float32)
        self.is_delta = False
        self.kd_default = True
        self.ks_default = True
        self.kg_default = True
        self.ka_default = True

        rgb_nodes = elem.findall("rgb")
        for rgb_node in rgb_nodes:
            name = rgb_node.get("name")
            if name is None: raise ValueError(f"RGB node in Blinn-phong BRDF <{elem.get('id')}> has empty name.")
            if name in BRDF_np.__all_albedo_name:
                self.k_d = rgb_parse(rgb_node)
                self.kd_default = False
            elif name in BRDF_np.__all_specular_name:
                self.k_s = rgb_parse(rgb_node)
                self.ks_default = False
            elif name in BRDF_np.__all_glossiness_name:
                self.k_g = rgb_parse(rgb_node)
                self.kg_default = False
            elif name in BRDF_np.__all_absorption_name:
                self.k_a = rgb_parse(rgb_node)
                self.ka_default = False
        if not no_setup:
            self.setup()

    def setup(self):
        if self.type not in BRDF_np.__type_mapping:
            raise NotImplementedError(f"Unknown BRDF type: {self.type}")
        self.type_id = BRDF_np.__type_mapping[self.type]
        if self.type_id == 2:
            if self.k_g.max() < 1e-4:       # glossiness (actually means roughness) in specular BRDF being too "small"
                self.is_delta = True
        elif self.type_id == 5:             # precomputed coefficient for Frensel Blend BRDF
            self.k_g[2] = np.sqrt((self.k_g[0] + 1) * (self.k_g[1] + 1)) / (8. * np.pi)

    def export(self):
        if self.type_id == -1:
            raise ValueError("It seems that this BRDF is not properly initialized with type_id = -1")
        return BRDF(
            _type = self.type_id, is_delta = self.is_delta, 
            k_d = vec3(self.k_d), k_s = vec3(self.k_s), k_g = vec3(self.k_g), k_a = vec3(self.k_a),
            mean = vec4([self.k_d.mean(), self.k_s.mean(), self.k_g.mean(), self.k_a.mean()])
        )
    
    def __repr__(self) -> str:
        return f"<{self.type.capitalize()} BRDF, default:[{int(self.kd_default), int(self.ks_default), int(self.kg_default), int(self.ka_default)}]>"

@ti.dataclass
class BRDF:
    """
        Taichi exported struct for unified BRDF storage
    """
    _type:      int
    is_delta:   int          # whether the BRDF is Dirac-delta-like
    k_d:        vec3            # diffusive coefficient (albedo)
    k_s:        vec3            # specular coefficient
    k_g:        vec3            # glossiness coefficient
    k_a:        vec3            # absorption coefficient
    mean:       vec4
    
    # ======================= Blinn-Phong ========================
    @ti.func
    def eval_blinn_phong(self, ray_in: vec3, ray_out: vec3, normal: vec3):
        """
            Normally, ray in is along the opposite direction of normal
            Attention: ray_in (in backward tracing) is actually out-going direction (in forward tracing)
            therefore, cosine term is related to ray_out
        """
        half_way = (ray_out - ray_in)
        if ti.abs(half_way).max() > EPS:
            half_way = half_way.normalized()
        else:
            half_way.fill(0.0)
        dot_clamp = ti.max(0.0, tm.dot(half_way, normal))
        glossy = tm.pow(dot_clamp, self.k_g)
        cosine_term = tm.max(0.0, tm.dot(normal, ray_out))
        # A modified Phong model (k_d + k_s should be smaller than 1, otherwise not physically plausible)
        return (self.k_d + self.k_s * (0.5 * (self.k_g + 2.0) * glossy)) * INV_PI * cosine_term

    @ti.func
    def sample_blinn_phong(self, incid: vec3, normal: vec3):
        local_new_dir, pdf = cosine_hemisphere()
        ray_out_d, _ = delocalize_rotate(normal, local_new_dir)
        spec = self.eval_blinn_phong(incid, ray_out_d, normal)
        return ray_out_d, spec, pdf

    # ======================  Modified Phong =======================
    @ti.func
    def eval_mod_phong(self, ray_in: vec3, ray_out: vec3, normal: vec3):
        dot_normal = tm.dot(normal, ray_out)
        spec = vec3([0, 0, 0])
        if dot_normal > 0.0:        # Phong model - specular part
            reflect_d = (2 * normal * dot_normal - ray_out).normalized()
            dot_view = ti.max(0.0, -tm.dot(ray_in, reflect_d))      # ray_in is on the opposite dir of reflected dir
            glossy = tm.pow(dot_view, self.k_g) * self.k_s
            spec = 0.5 * (self.k_g + 2.) * glossy * INV_PI * dot_normal
        return spec 

    @ti.func
    def sample_mod_phong(self, incid: vec3, normal: vec3):
        # Sampling is more complicated
        eps = ti.random(float)
        ray_out_d = vec3([0, 1, 0])
        spec = vec3([0, 0, 0])
        pdf = self.k_d.max()
        if eps < pdf:                       # diffusive sampling
            ray_out_d, spec, lmbt_pdf = self.sample_lambertian(normal)
            pdf *= lmbt_pdf
        elif eps < pdf + self.k_s.max():    # specular sampling
            local_new_dir, pdf = mod_phong_hemisphere(self.mean[2])
            reflect_view = (-2 * normal * tm.dot(incid, normal) + incid).normalized()
            ray_out_d, _ = delocalize_rotate(reflect_view, local_new_dir)
            spec = self.eval_mod_phong(incid, ray_out_d, normal)
            pdf *= self.k_s.max()
        else:                               # zero contribution
            # it doesn't matter even we don't return a valid ray_out_d
            # since returned spec here is 0, contribution will be 0 and the ray will be terminated by RR or cut-off
            pdf = 1. - pdf - self.k_s.max()     # seems to be absorbed
        # Sample around reflected view dir (while blinn-phong samples around normal)
        return ray_out_d, spec, pdf
    
    # ======================= Frensel-Blend =======================
    """
        For Frensel Blend (by Ashikhmin and Shirley 2002), n_u and n_v will be stored in k_g
        since k_g will not be used, k_d and k_s preserve their original meaning
    """

    @ti.func
    def frensel_blend_dir(self, incid: vec3, half: vec3, normal: vec3, power_coeff: float):
        reflected, dot_incid = inci_reflect_dir(incid, half)
        half_pdf = self.k_g[2] * tm.pow(tm.dot(half, normal), power_coeff)
        pdf = half_pdf / ti.max(ti.abs(dot_incid), EPS)
        valid_sample = tm.dot(normal, reflected) > 0.
        return reflected, pdf, valid_sample
    
    @ti.func
    def frensel_cos2_sin2(self, half_vec: vec3, normal: vec3, R: mat3, dot_half: float):
        transed_x = (R @ vec3([1, 0, 0])).normalized()
        cos_phi2  = tm.dot(transed_x, (half_vec - dot_half * normal).normalized()) ** 2       # azimuth angle of half vector 
        return cos_phi2, 1. - cos_phi2

    @ti.func
    def eval_frensel_blend(self, ray_in: vec3, ray_out: vec3, normal: vec3, R: mat3):
        # specular part, note that ray out is actually incident light in forward tracing
        half_vec = (ray_out - ray_in)
        dot_out  = tm.dot(normal, ray_out)
        spec = vec3([0, 0, 0])
        if dot_out > 0. and ti.abs(half_vec).max() > 1e-4:  # ray_in and ray_out not on the exact opposite direction
            half_vec = half_vec.normalized()
            dot_in   = -tm.dot(normal, ray_in)              # incident dot should always be positive (otherwise it won't hit this point)
            dot_half = ti.abs(tm.dot(normal, half_vec))
            dot_hk   = ti.abs(tm.dot(half_vec, ray_out))
            frensel  = schlick_frensel(self.k_s, dot_hk)
            cos_phi2, sin_phi2 = self.frensel_cos2_sin2(half_vec, normal, R, dot_half)
            # k_g[2] should store sqrt((n_u + 1)(n_v + 1)) / 8pi
            denom = dot_hk * tm.max(dot_in, dot_out)
            specular = self.k_g[2] * tm.pow(dot_half, self.k_g[0] * cos_phi2 + self.k_g[1] * sin_phi2) * frensel / denom
            # diffusive part
            diffuse  = 28. / (23. * tm.pi) * self.k_d * (1. - self.k_s)
            pow5_in  = tm.pow(1. - dot_in / 2., 5)
            pow5_out = tm.pow(1. - dot_out / 2., 5)
            diffuse *= (1. - pow5_in) * (1. - pow5_out)
            spec = (specular + diffuse) * dot_out
        return spec

    @ti.func
    def sample_frensel_blend(self, incid: vec3, normal: vec3):
        local_new_dir, power_coeff = frensel_hemisphere(self.k_g[0], self.k_g[1])
        ray_half, R = delocalize_rotate(normal, local_new_dir)
        ray_out_d, pdf, is_valid = self.frensel_blend_dir(incid, ray_half, normal, power_coeff)
        spec = vec3([0, 0, 0])
        if is_valid:
            spec = self.eval_frensel_blend(incid, ray_out_d, normal, R)
        return ray_out_d, spec, pdf
    
    # ======================= Lambertian ========================
    @ti.func
    def eval_lambertian(self, ray_out: vec3, normal: vec3):
        cosine_term = tm.max(0.0, tm.dot(normal, ray_out))
        return self.k_d * INV_PI * cosine_term

    @ti.func
    def sample_lambertian(self, normal: vec3):
        local_new_dir, pdf = cosine_hemisphere()
        ray_out_d, _ = delocalize_rotate(normal, local_new_dir)
        spec = self.eval_lambertian(ray_out_d, normal)
        return ray_out_d, spec, pdf

    # ======================= Mirror-Specular ========================
    @ti.func
    def eval_specular(self, ray_in: vec3, ray_out: vec3, normal: vec3):
        """ Attention: ray_in (in backward tracing) is actually out-going direction (in forward tracing) """
        reflect_dir, _ = inci_reflect_dir(ray_in, normal)
        spec = vec3([0, 0, 0])
        if tm.dot(ray_out, reflect_dir) > 1 - 1e-4:
            spec = self.k_d
        return spec

    @ti.func
    def sample_specular(self, ray_in: vec3, normal: vec3):
        ray_out_d, _ = inci_reflect_dir(ray_in, normal)
        return ray_out_d, self.k_d, 1.0

    # ================================================================

    @ti.func
    def eval(self, incid: vec3, out: vec3, normal: vec3) -> vec3:
        """ Direct component reflectance """
        ret_spec = vec3([1, 1, 1])
        if self._type == 0:         # Blinn-Phong
            ret_spec = self.eval_blinn_phong(incid, out, normal)
        elif self._type == 1:       # Lambertian
            ret_spec = self.eval_lambertian(out, normal)
        elif self._type == 2:       # Specular
            ret_spec = self.eval_specular(incid, out, normal)
        elif self._type == 4:
            ret_spec = self.eval_mod_phong(incid, out, normal)
        elif self._type == 5:
            R = rotation_between(vec3([0, 1, 0]), normal)
            ret_spec = self.eval_frensel_blend(incid, out, normal, R)
        else:
            print(f"Warnning: unknown or unsupported BRDF type: {self._type} during evaluation.")
        return ret_spec

    @ti.func
    def sample_new_rays(self, incid: vec3, normal: vec3):
        """
            All the sampling function will return: (1) new ray (direction) \\
            (2) rendering equation transfer term (BRDF * cos term) (3) PDF
        """
        ret_dir  = vec3([0, 1, 0])
        ret_spec = vec3([1, 1, 1])
        pdf      = 1.0
        if self._type == 0:         # Blinn-Phong
            ret_dir, ret_spec, pdf = self.sample_blinn_phong(incid, normal)
        elif self._type == 1:       # Lambertian
            ret_dir, ret_spec, pdf = self.sample_lambertian(normal)
        elif self._type == 2:       # Specular
            ret_dir, ret_spec, pdf = self.sample_specular(incid, normal)
        elif self._type == 4:       # Modified-Phong
            ret_dir, ret_spec, pdf = self.sample_mod_phong(incid, normal)
        elif self._type == 5:       # Frensel-Blend
            ret_dir, ret_spec, pdf = self.sample_frensel_blend(incid, normal)
        else:
            print(f"Warnning: unknown or unsupported BRDF type: {self._type} during evaluation.")
        return ret_dir, ret_spec, pdf

    @ti.func
    def get_pdf(self, outdir: vec3, normal: vec3, incid: vec3):
        """ 
            Solid angle PDF for a specific incident direction - BRDF sampling
            Some PDF has nothing to do with backward incid (from eye to the surface), like diffusive 
            This PDF is actually the PDF of cosine-weighted term * BRDF function value
        """
        pdf = 0.0
        dot_outdir = tm.dot(normal, outdir)
        if self._type == 0:
            pdf = tm.max(dot_outdir, 0.0) * INV_PI      # dot is cosine term
        elif self._type == 1:
            pdf = tm.max(dot_outdir, 0.0) * INV_PI
        elif self._type == 4:
            if dot_outdir > 0.0:
                glossiness      = self.mean[2]
                reflect_view, _ = inci_reflect_dir(incid, normal)
                dot_ref_out     = tm.max(0., tm.dot(reflect_view, outdir))
                diffuse_pdf     = tm.max(dot_outdir, 0.0) * INV_PI
                specular_pdf    = 0.5 * (glossiness + 1.) * INV_PI * tm.pow(dot_ref_out, glossiness)
                pdf = self.k_d.max() * diffuse_pdf + self.k_s.max() * specular_pdf
        elif self._type == 5:
            if dot_outdir > 0.0: 
                half_vec = (outdir - incid).normalized()
                dot_half = tm.dot(half_vec, normal)
                R = rotation_between(vec3([0, 1, 0]), normal)
                cos_phi2, sin_phi2 = self.frensel_cos2_sin2(half_vec, normal, R, dot_half)
                pdf = self.k_g[2] * tm.pow(dot_half, self.k_g[0] * cos_phi2 + self.k_g[1] * sin_phi2) * 4.
        return pdf