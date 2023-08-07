"""
    All the BRDFs are here, note that only three kinds of simple BRDF are supported
    Blinn-Phong / Lambertian / Mirror specular / Modified Phong / Fresnel Blend
    @author: Qianyue He
    @date: 2023-1-23
"""

__ENABLE_MICROFACET__ = False

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
from parsers.general_parser import rgb_parse
from renderer.constants import INV_PI, ZERO_V3, DEG2RAD, BRDFTag

if ti.static(__ENABLE_MICROFACET__):
    from sampler.microfacet import *

from rich.console import Console
CONSOLE = Console(width = 128)

__all__ = ['BRDF_np', 'BRDF']

EPS = 1e-7

class BRDF_np:
    """
        BRDF base-class, 
        @author: Qianyue He
        @date: 2023-1-23
    """
    __all_albedo_name       = {"reflectance", "albedo", "k_d"}
    __all_glossiness_name   = {"glossiness", "shininess", "roughness", "sigma", "k_g"}
    __all_specular_name     = {"specular", "ref_ior", "k_s"}
    # Attention: microfacet support will not be added recently
    __type_mapping          = {"blinn-phong": 0, "lambertian": 1, "specular": 2, "microfacet": 3, 
                               "mod-phong": 4, "fresnel-blend": 5, "oren-nayar": 6, "thin-coat": 7}
    
    def __init__(self, elem: xet.Element, no_setup = False):
        self.type: str = elem.get("type")
        self.type_id = BRDF_np.__type_mapping.get(self.type, -1)
        self.id: str = elem.get("id")
        self.k_d = np.ones(3, np.float32)
        self.k_s = np.zeros(3, np.float32)
        self.k_g = np.ones(3, np.float32)
        self.is_delta = False
        self.kd_default = True
        self.ks_default = True
        self.kg_default = True
        self.uv_coords = None
        if not __ENABLE_MICROFACET__ and self.type_id == BRDFTag.MICROFACET:
            CONSOLE.log(f"[yellow]Warning: [/yellow]BRDF <{self.id}> is microfacet while microfacet BRDF is not enabled.")
            CONSOLE.log(f"Falling back to Lambertian BRDF for BRDF <{self.id}>")
            CONSOLE.log("Try setting `__ENABLE_MICROFACET__ = True`, while this might slow down JIT compilation.")
            self.type = "lambertian"
            self.type_id = BRDFTag.LAMBERTIAN

        texture_nodes = elem.findall("texture")
        if len(texture_nodes) > 1:
            CONSOLE.log(f"[yellow]Warning: [/yellow]Only one texture is supported in a BR(S)DF. Some textures might be shadowed for <{self.id}>.")
        rgb_nodes = elem.findall("rgb")
        if len(rgb_nodes) == 0 and len(texture_nodes) == 0:
            CONSOLE.log(f"[yellow]Warning: [/yellow]BSDF <{self.id}> has no surface color / textures defined.")
        for rgb_node in rgb_nodes:
            name = rgb_node.get("name")
            if name is None: 
                raise ValueError(f"RGB node in BR(S)DF <{elem.get('id')}> has empty name.")
            if name in BRDF_np.__all_albedo_name:
                self.k_d = rgb_parse(rgb_node)
                self.kd_default = False
            elif name in BRDF_np.__all_specular_name:
                self.k_s = rgb_parse(rgb_node)
                self.ks_default = False
            elif name in BRDF_np.__all_glossiness_name:
                self.k_g = rgb_parse(rgb_node)
                self.kg_default = False
                if self.type_id == BRDFTag.MICROFACET:
                    if name in BRDF_np.__all_glossiness_name - {"roughness"}:
                        CONSOLE.log(f"[yellow]Warning: [/yellow]Microfacet BRDF model should have attribute."
                                    f"'roughness' instead of '{name}' as k_g.")
                    elif name in BRDF_np.__all_specular_name - {"ref_ior"}:
                        CONSOLE.log(f"[yellow]Warning: [/yellow]Microfacet BRDF model should have attribute."
                                    f"'ref_ior' (Index of Refraction) instead of '{name}' as k_s.")
                elif self.type_id in {BRDFTag.OREN_NAYAR, BRDFTag.THIN_COAT}:
                    if name in BRDF_np.__all_glossiness_name - {"sigma"}:
                        CONSOLE.log(f"[yellow]Warning: [/yellow]Oren-Nayar based BRDF models should have attribute."
                                    f"'sigma' instead of '{name}' as k_g.")
                if name == "roughness":
                    # convert from roughness to alpha values (for now, only used in microfacet BRDF)
                    if (self.k_g > 1).any() or (self.k_g < 0).any():
                        CONSOLE.log(f"[yellow]Warning: [/yellow]roughness for microfacet BRDF <{self.id}>"
                                    " should be in range [0, 1]. Clamped to [0, 1].")
                        self.k_g = self.k_g.clip(0, 1)
                    self.k_g = BRDF_np.roughness_to_alpha(self.k_g)
                elif name == "sigma":
                    # convert from spherical guassian parameter to Oren-Nayar parameter
                    sigma = self.k_g[0] * DEG2RAD
                    sigma2 = sigma * sigma
                    self.k_g[0] = 1 - (sigma2 / (2 * (sigma2 + 0.33)))
                    self.k_g[1] = 0.45 * sigma2 / (sigma2 + 0.09)
                    self.k_g[2] = max(1., self.k_g[2])                      # IOR for thin-coat
                    
        if not no_setup:
            self.setup()

    @staticmethod
    def roughness_to_alpha(roughness: np.ndarray) -> np.ndarray:
        """ From PBRT-v3 core/microfacet.h: TrowbridgeReitzDistribution::RoughnessToAlpha """
        x = np.log(np.maximum(roughness, 1e-3))
        return 1.62142 + 0.819955 * x + 0.1734 * x * x + 0.0171201 * (x ** 3) + \
               0.000640711 * (x ** 4)

    def setup(self):
        if self.type not in BRDF_np.__type_mapping:
            raise NotImplementedError(f"Unknown BRDF type: {self.type}")
        if self.type_id == BRDFTag.SPECULAR:
            self.is_delta = True
        elif self.type_id == BRDFTag.FRESNEL_BLEND:             # precomputed coefficient for Fresnel Blend BRDF
            self.k_g[2] = np.sqrt((self.k_g[0] + 1) * (self.k_g[1] + 1)) / (8. * np.pi)

    def export(self):
        if self.type_id == -1:
            raise ValueError("It seems that this BRDF is not properly initialized with type_id = -1")
        return BRDF(
            _type = self.type_id, is_delta = self.is_delta, 
            k_d = vec3(self.k_d), k_s = vec3(self.k_s), k_g = vec3(self.k_g),
            mean = vec3([self.k_d.mean(), self.k_s.mean(), self.k_g.mean()])
        )
    
    def __repr__(self) -> str:
        return f"<{self.type.capitalize()} BRDF, default:[{int(self.kd_default), int(self.ks_default), int(self.kg_default)}]>"

# ============================================================================

# =================== Thin coat (Fresnel coating for plastic material) ================
# ============================================================================

@ti.dataclass
class BRDF:
    """
        Taichi exported struct for unified BRDF storage
    """
    _type:      int
    is_delta:   int          # whether the BRDF is Dirac-delta-like
    """ Note that thin-coat can be delta / non-delta at the same time (have specular + diffuse part)"""
    k_d:        vec3            # diffusive coefficient (albedo)
    k_s:        vec3            # specular coefficient
    k_g:        vec3            # glossiness coefficient
    mean:       vec3
    """ For some BRDFs, like modified phong, mean is exactly what it means
        But for some others, like Microfacet (T-S / O-N), mean carries the meaning of 
        incident IOR (mean[0]) and transmission IOR (mean[1])
    """
    
    # ======================= Blinn-Phong ========================
    @ti.func
    def eval_blinn_phong(self, it:ti.template(), ray_in: vec3, ray_out: vec3):
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
        dot_clamp = ti.max(0.0, tm.dot(half_way, it.n_s))
        glossy = tm.pow(dot_clamp, self.k_g)
        cosine_term = ti.max(0.0, tm.dot(it.n_s, ray_out))
        # A modified Phong model (k_d + k_s should be smaller than 1, otherwise not physically plausible)
        diffuse_color = ti.select(it.is_tex_invalid(), self.k_d, it.tex)
        return (diffuse_color + self.k_s * (0.5 * (self.k_g + 2.0) * glossy)) * INV_PI * cosine_term

    @ti.func
    def sample_blinn_phong(self, it:ti.template(), incid: vec3):
        local_new_dir, pdf = cosine_hemisphere()
        ray_out_d, _ = delocalize_rotate(it.n_s, local_new_dir)
        spec = self.eval_blinn_phong(it, incid, ray_out_d)
        return ray_out_d, spec, pdf

    # ======================  Modified Phong =======================
    """ Modified Phong BRDF
        Lafortune & Willems, using the modified phong reflectance model for physically based rendering, 1994
    """

    @ti.func
    def eval_mod_phong(self, it: ti.template(), ray_in: vec3, ray_out: vec3):
        dot_normal = tm.dot(it.n_s, ray_out)
        spec = vec3([0, 0, 0])
        if dot_normal > 0.0:        # Phong model - specular part
            reflect_d = (2 * it.n_s * dot_normal - ray_out).normalized()
            dot_view = ti.max(0.0, -tm.dot(ray_in, reflect_d))      # ray_in is on the opposite dir of reflected dir
            glossy = tm.pow(dot_view, self.k_g) * self.k_s
            spec = 0.5 * (self.k_g + 2.) * glossy * INV_PI * dot_normal
            spec += self.eval_lambertian(it, it.n_s, ray_out)
        return spec 

    @ti.func
    def sample_mod_phong(self, it: ti.template(), incid: vec3):
        # Sampling is more complicated
        eps = ti.random(float)
        ray_out_d = vec3([0, 1, 0])
        spec = vec3([0, 0, 0])
        pdf = ti.select(it.is_tex_invalid(), self.k_d, it.tex).max()    # the max element of diffuse color is the pdf of sampling diffusive component
        if eps < pdf:                       # diffusive sampling
            ray_out_d, spec, lmbt_pdf = self.sample_lambertian(it, it.n_s)   # we use "texture" anyway, since it is converted
            pdf *= lmbt_pdf
        elif eps < pdf + self.k_s.max():    # specular sampling
            local_new_dir, pdf = mod_phong_hemisphere(self.mean[2])
            reflect_view = (-2 * it.n_s * tm.dot(incid, it.n_s) + incid).normalized()
            ray_out_d, _ = delocalize_rotate(reflect_view, local_new_dir)
            spec = self.eval_mod_phong(it, incid, ray_out_d)
            pdf *= self.k_s.max()
        else:                               # zero contribution
            # it doesn't matter even we don't return a valid ray_out_d
            # since returned spec here is 0, contribution will be 0 and the ray will be terminated by RR or cut-off
            pdf = 1. - pdf - self.k_s.max()     # seems to be absorbed
        # Sample around reflected view dir (while blinn-phong samples around normal)
        return ray_out_d, spec, pdf
    
    # ======================= Fresnel-Blend =======================
    """
        For Fresnel Blend (by Ashikhmin and Shirley 2002), n_u and n_v will be stored in k_g
        since k_g will not be used, k_d and k_s preserve their original meaning
    """

    @ti.func
    def fresnel_blend_dir(self, incid: vec3, half: vec3, normal: vec3, power_coeff: float):
        reflected, dot_incid = inci_reflect_dir(incid, half)
        half_pdf = self.k_g[2] * tm.pow(tm.dot(half, normal), power_coeff)
        pdf = half_pdf / ti.max(ti.abs(dot_incid), EPS)

        valid_sample = tm.dot(normal, reflected) > 0.
        return reflected, pdf, valid_sample
    
    @ti.func
    def fresnel_cos2_sin2(self, half_vec: vec3, normal: vec3, R: mat3, dot_half: float):
        transed_x = (R @ vec3([1, 0, 0]))
        cos_phi2  = tm.dot(transed_x, (half_vec - dot_half * normal).normalized()) ** 2       # azimuth angle of half vector 
        return cos_phi2, 1. - cos_phi2

    @ti.func
    def eval_fresnel_blend(self, it:ti.template(), ray_in: vec3, ray_out: vec3, R: mat3):
        # specular part, note that ray out is actually incident light in forward tracing
        half_vec = (ray_out - ray_in)
        dot_out  = tm.dot(it.n_s, ray_out)
        spec = vec3([0, 0, 0])
        if dot_out > 0. and ti.abs(half_vec).max() > 1e-4:  # ray_in and ray_out not on the exact opposite direction
            half_vec = half_vec.normalized()
            dot_in   = -tm.dot(it.n_s, ray_in)              # incident dot should always be positive (otherwise it won't hit this point)
            dot_half = ti.abs(tm.dot(it.n_s, half_vec))
            dot_hk   = ti.abs(tm.dot(half_vec, ray_out))
            fresnel  = schlick_fresnel(self.k_s, dot_hk)
            cos_phi2, sin_phi2 = self.fresnel_cos2_sin2(half_vec, it.n_s, R, dot_half)
            # k_g[2] should store sqrt((n_u + 1)(n_v + 1)) / 8pi
            denom = dot_hk * tm.max(dot_in, dot_out)
            specular = self.k_g[2] * tm.pow(dot_half, self.k_g[0] * cos_phi2 + self.k_g[1] * sin_phi2) * fresnel / denom
            # diffusive part
            diffuse_color = ti.select(it.is_tex_invalid(), self.k_d, it.tex)
            diffuse  = 28. / (23. * tm.pi) * diffuse_color * (1. - self.k_s)
            pow5_in  = tm.pow(1. - dot_in / 2., 5)
            pow5_out = tm.pow(1. - dot_out / 2., 5)
            diffuse *= (1. - pow5_in) * (1. - pow5_out)
            spec = (specular + diffuse) * dot_out
        return spec

    @ti.func
    def sample_fresnel_blend(self, it: ti.template(), incid: vec3):
        local_new_dir, power_coeff = fresnel_hemisphere(self.k_g[0], self.k_g[1])
        ray_half, R = delocalize_rotate(it.n_s, local_new_dir)
        ray_out_d, pdf, is_valid = self.fresnel_blend_dir(incid, ray_half, it.n_s, power_coeff)
        if ti.random(float) > 0.5:
            ray_out_d, _s, _p = self.sample_lambertian(it, it.n_s)
        pdf = 0.5 * (pdf + ti.abs(tm.dot(ray_out_d, it.n_s)) * INV_PI)
        spec = ti.select(is_valid, self.eval_fresnel_blend(it, incid, ray_out_d, R), ZERO_V3)
        return ray_out_d, spec, pdf
    
    # ======================= Lambertian ========================
    # We keep the 'normal' parameter for Lambertian and Specular, for they will be used in microfacet models
    @ti.func
    def eval_lambertian(self, it: ti.template(), normal: vec3, ray_out: vec3):
        cosine_term = tm.max(0.0, tm.dot(normal, ray_out))
        diffuse_color = ti.select(it.is_tex_invalid(), self.k_d, it.tex)
        return diffuse_color * INV_PI * cosine_term
    
    @ti.func
    def sample_lambertian(self, it: ti.template(), normal: vec3):
        local_new_dir, pdf = cosine_hemisphere()
        ray_out_d, _ = delocalize_rotate(normal, local_new_dir)
        spec = self.eval_lambertian(it, normal, ray_out_d)
        return ray_out_d, spec, pdf

    # ======================= Mirror-Specular ========================
    @ti.func
    def sample_specular(self, it: ti.template(), ray_in: vec3, normal: vec3):
        ray_out_d, _ = inci_reflect_dir(ray_in, normal)
        return ray_out_d, ti.select(it.is_tex_invalid(), self.k_d, it.tex), 1.0
    
    # ================================================================

    # ======================= Oren-Nayar callable member (PBR diffuse) =============================
    @ti.func
    def eval_oren_nayar(self, it:ti.template(), ray_in: vec3, ray_out: vec3):
        raw_wi = convert_to_raw(-ray_in, it.n_s)
        raw_wo = convert_to_raw(ray_out, it.n_s)
        sin_theta_i = raw_wi[1]
        sin_theta_o = raw_wo[1]
        max_cos = 0.
        if sin_theta_i > 1e-5 and sin_theta_o > 1e-5:
            cos_phi_i = raw_wi[2]
            sin_phi_i = raw_wi[3] 

            cos_phi_o = raw_wo[2]
            sin_phi_o = raw_wo[3] 
            d_cos = cos_phi_i * cos_phi_o + sin_phi_i * sin_phi_o
            max_cos = ti.max(0., d_cos)

        sin_alpha = 0.
        tan_beta = 0.
        abs_cos_wi = ti.abs(raw_wi[0])
        abs_cos_wo = ti.abs(raw_wo[0])

        if abs_cos_wi > abs_cos_wo:
            sin_alpha = sin_theta_o
            tan_beta = sin_theta_i / abs_cos_wi
        else:
            sin_alpha = sin_theta_i
            tan_beta = sin_theta_o / abs_cos_wo
        diffuse_color = ti.select(it.is_tex_invalid(), self.k_d, it.tex)
        # ti.abs(raw_wo[0]) is cosine term
        spec = diffuse_color * INV_PI * (self.k_g[0] + self.k_g[1] * max_cos * sin_alpha * tan_beta) * ti.abs(raw_wo[0])
        return spec

    # ============================================================================

    # ======================= Thin coat (Fresnel coating for plastic material) =====================

    @ti.func
    def sample_thin_coat(self, it: ti.template(), incid: vec3):
        """ Sampling the Fresnel coating 
            NOTE that: the IOR for Fresnel coating does not account for the IOR for the outside medium
            for example, if we have glass outside (1.5), this would mean we will have 1.5 * ior for the inside 
        """
        pdf = 1.0
        spec = ZERO_V3
        ray_out_d = vec3([0, 1, 0])

        dot_normal = tm.dot(incid, it.n_s)
        # We will not have total reflection for incident ray
        refra_in, cos_r2 = snell_refraction(incid, it.n_s, dot_normal, 1.0, self.k_g[2])
        in_ref_F = fresnel_equation(1., self.k_g[0], ti.abs(dot_normal), ti.sqrt(cos_r2))
        """ There are two Fresnel term sampling:
            - sample: direct reflection
            - sample: total internal reflection in the coating layer - absorbed
        """
        is_specular = False
        if ti.random(float) > in_ref_F:       # refracting into the thin layer
            # Lambertian sampling
            local_new_dir, pdf = cosine_hemisphere()
            ray_out_d, _ = delocalize_rotate(it.n_s, local_new_dir)
            dot_out = tm.dot(ray_out_d, it.n_s)
            if not is_total_reflection(dot_out, self.k_g[2], 1.0):      # total reflection - set spec to 0
                # sample the second Fresnel term: whether we will have a total reflection?
                refra_out, cos_r2 = snell_refraction(ray_out_d, it.n_s, dot_out, self.k_g[2], 1.0)
                out_ref_F = fresnel_equation(self.k_g[2], 1., ti.abs(dot_out), ti.sqrt(cos_r2))
                pdf *= (1. - in_ref_F)
                ray_out_d = refra_out
                # Oren-Nayar evaluation
                spec = self.eval_oren_nayar(it, refra_in, ray_out_d)
                spec *= (1. - in_ref_F) * (1. - out_ref_F)
        else:                               # reflection
            # sample specular, we should tag it
            spec = self.k_s * in_ref_F
            ray_out_d, _ = inci_reflect_dir(incid, it.n_s)
            pdf = in_ref_F
            is_specular = True
        return ray_out_d, spec, pdf, is_specular

    @ti.func
    def eval_thin_coating(self, it:ti.template(), ray_in: vec3, ray_out: vec3):
        # input Fresnel evaluation
        ret_spec = ZERO_V3
        reflect, _ = inci_reflect_dir(ray_in, it.n_s)
        dot_in = tm.dot(ray_in, it.n_s)
        # We will not have total reflection for incident ray
        refra_in, cos_r2 = snell_refraction(ray_in, it.n_s, dot_in, 1.0, self.k_g[2])
        in_ref_F = fresnel_equation(1., self.k_g[2], ti.abs(dot_in), ti.sqrt(cos_r2))
        if ti.abs(tm.dot(ray_out, reflect)) > (1. - 1e-4):
            ret_spec = self.k_s * in_ref_F
        else:
            # output Fresnel evaluation
            dot_out = tm.dot(ray_out, it.n_s)
            refra_out, cos_r2 = snell_refraction(ray_out, it.n_s, dot_out, 1.0, self.k_g[2])    # always valid
            out_ref_F = fresnel_equation(1.0, self.k_g[2], ti.abs(dot_out), ti.sqrt(cos_r2))
            ret_spec = self.eval_oren_nayar(it, refra_in, refra_out) * ((1. - out_ref_F) * (1. - in_ref_F))
        return ret_spec
    
    @ti.func
    def thin_coat_fresnel(self, it: ti.template(), ray_in: vec3) -> float:
        """ Evaluating the Fresnel coating: mirror specular + Oren-Nayar diffuse 
            note that evaluation for mirror specular part is always 0

            in_ref_F can be precomputed (during sampling)
        """
        # input Fresnel evaluation
        dot_in = tm.dot(ray_in, it.n_s)
        # We will not have total reflection for incident ray
        ratio    = 1.0 / self.k_g[2]
        cos_r2   = 1. - ti.pow(ratio, 2) * (1. - ti.pow(dot_in, 2))        # refraction angle cosine
        in_ref_F = fresnel_equation(1., self.k_g[2], ti.abs(dot_in), ti.sqrt(cos_r2))
        return in_ref_F

    # ============================================================================

    # ======================= Microfacet Torrance–Sparrow (PBR Glossy) =============================

    if ti.static(__ENABLE_MICROFACET__):
        @ti.func
        def sample_microfacet(self, it: ti.template(), incid: vec3):
            """ Note that for Torrance Sparrow and Oren-Nayar, alpha is converted from roughness and stored in k_g
                Also, incid ray points inwards
            """
            # It is pretty strange that sometimes we need to invert the direction
            local_wh, raw_vec = trow_reitz_sample_wh(incid, it.n_s, self.k_g[0], self.k_g[1])
            half_vector, _ = delocalize_rotate(it.n_s, local_wh)
            dot_val = -tm.dot(incid, half_vector)
            ret_spec = ZERO_V3
            pdf = 1.0
            ray_out_d = vec3([0, 1, 0])
            if dot_val > 0:
                ray_out_d, _ = inci_reflect_dir(incid, half_vector)
                # it.n_s dot incid should be negative
                cos_theta_o = tm.dot(it.n_s, ray_out_d)
                cos_theta_i = tm.dot(it.n_s, incid)

                if cos_theta_o * cos_theta_i < 0:
                    cos_theta_i = ti.abs(cos_theta_i)
                    cos_theta_o = ti.abs(cos_theta_o)
                    if cos_theta_o > EPS and cos_theta_i > EPS:
                        ret_spec = self.eval_microfacet_with_raw(it, half_vector, raw_vec, incid, ray_out_d)
                        ret_spec /= (4. * cos_theta_o * cos_theta_i)
                        pdf = trow_reitz_pdf(-incid, half_vector, self.k_g, it.n_s)
                        pdf /= 4. * dot_val
            return ray_out_d, ret_spec, pdf

        @ti.func
        def eval_microfacet_with_raw(self, it:ti.template(), wh: vec3, raw_vec: vec4, ray_in: vec3, ray_out: vec3):
            """ The output of this function is not divided by 4 * cos_i * cos_o 
                Remember, all eval funcs will carry the cosine-term (except delta BRDF) 
            """
            ret_spec = ZERO_V3
            if ti.abs(wh[0]) > EPS or ti.abs(wh[1]) > EPS or ti.abs(wh[2]) > EPS:
                wh = wh.normalized()
                dot_hk = tm.dot(wh, ray_out)
                fresnel = fresnel_eval(dot_hk, self.k_s[0], self.k_s[1])
                diffuse_color = ti.select(it.is_tex_invalid(), self.k_d, it.tex)
                cosine_term = ti.abs(tm.dot(it.n_s, ray_out))
                ret_spec = diffuse_color * trow_reitz_D(raw_vec, self.k_g) * \
                    trow_reitz_G(-ray_in, ray_out, self.k_g, it.n_s) * fresnel * cosine_term
            return ret_spec

        @ti.func
        def eval_microfacet(self, it:ti.template(), ray_in: vec3, ray_out: vec3):
            ret_spec = ZERO_V3
            cos_theta_o = tm.dot(it.n_s, ray_out)
            cos_theta_i = tm.dot(it.n_s, ray_in)
            cos_mult = cos_theta_o * cos_theta_i
            if cos_mult < 0.:
                wh = (ray_out - ray_in).normalized()
                raw_vec = convert_to_raw(wh, it.n_s)
                ret_spec = self.eval_microfacet_with_raw(it, wh, raw_vec, ray_in, ray_out)
                ret_spec /= -4. * cos_mult
            return ret_spec 
    else:
        """ Since microfacet functions can slow down JIT compilation (multiple times)
            Normally I do not enable microfacet BRDF, so use FresnelBlend / Modified Phong instead
        """
        @ti.func
        def sample_microfacet(self, _it: ti.template(), _incid: vec3):
            return vec3([0, 1, 0]), ZERO_V3, 1.0
        
        @ti.func
        def eval_microfacet_with_raw(self, _it:ti.template(), _wh: vec3, _raw_vec: vec4, _ray_in: vec3, _ray_out: vec3):
            return ZERO_V3
        
        @ti.func
        def eval_microfacet(self, _it:ti.template(), _ray_in: vec3, _ray_out: vec3):
            return ZERO_V3

    # ===================================================================================

    @ti.func
    def eval(self, it: ti.template(), incid: vec3, out: vec3) -> vec3:
        """ Direct component reflectance
            Every evaluation function does not output cosine weighted BSDF now
        """
        ret_spec = vec3([0, 0, 0])
        # For reflection, incident (in reverse direction) & outdir should be in the same hemisphere defined by the normal 
        if tm.dot(incid, it.n_g) * tm.dot(out, it.n_g) < 0:
            if self._type == BRDFTag.BLING_PHONG:         # Blinn-Phong
                ret_spec = self.eval_blinn_phong(it, incid, out)
            elif self._type == BRDFTag.LAMBERTIAN:       # Lambertian
                ret_spec = self.eval_lambertian(it, it.n_s, out)
            elif self._type == BRDFTag.MOD_PHONG:
                ret_spec = self.eval_mod_phong(it, incid, out)
            elif self._type == BRDFTag.FRESNEL_BLEND:
                R = rotation_between(vec3([0, 1, 0]), it.n_s)
                ret_spec = self.eval_fresnel_blend(it, incid, out, R)
            elif self._type == BRDFTag.OREN_NAYAR:
                ret_spec = self.eval_oren_nayar(it, incid, out)
            elif self._type == BRDFTag.THIN_COAT:
                ret_spec = self.eval_thin_coating(it, incid, out)
            elif self._type == BRDFTag.MICROFACET:
                ret_spec = self.eval_microfacet(it, incid, out)
        return ret_spec
    
    @ti.func
    def sample_new_rays(self, it:ti.template(), incid: vec3):
        """
            All the sampling function will return: (1) new ray (direction) \\
            (2) rendering equation transfer term (BRDF * cos term) (3) PDF
            mode for separating camera / light transport cosine term
        """

        # TODO: add enum for different types of BRDF
        ret_dir  = vec3([0, 1, 0])
        ret_spec = vec3([1, 1, 1])
        pdf      = 1.0
        is_specular = False 
        if self._type == BRDFTag.BLING_PHONG:         # bling-phong glossy sampling
            ret_dir, ret_spec, pdf = self.sample_blinn_phong(it, incid)
        elif self._type == BRDFTag.LAMBERTIAN or self._type == BRDFTag.OREN_NAYAR:       # Lambertian
            ret_dir, ret_spec, pdf = self.sample_lambertian(it, it.n_s)
        elif self._type == BRDFTag.SPECULAR:         # Specular - specular has no cosine attenuation
            ret_dir, ret_spec, pdf = self.sample_specular(it, incid, it.n_s)
        elif self._type == BRDFTag.THIN_COAT:
            ret_dir, ret_spec, pdf, is_specular = self.sample_thin_coat(it, incid)
        elif self._type == BRDFTag.MOD_PHONG:        # Modified-Phong
            ret_dir, ret_spec, pdf = self.sample_mod_phong(it, incid)
        elif self._type == BRDFTag.FRESNEL_BLEND:    # Fresnel-Blend
            ret_dir, ret_spec, pdf = self.sample_fresnel_blend(it, incid)
        elif self._type == BRDFTag.MICROFACET:       # Microfacet
            ret_dir, ret_spec, pdf = self.sample_microfacet(it, incid)
        else:
            print(f"Warnning: unknown or unsupported BRDF type: {self._type} during sampling.")
        # Prevent shading normal from light leaking or accidental shadowing
        ret_dot = tm.dot(ret_dir, it.n_g)
        ret_spec = ti.select(ret_dot > 0, ret_spec, 0.)
        return ret_dir, ret_spec, pdf, is_specular

    @ti.func
    def get_pdf(self, it: ti.template(), outdir: vec3, incid: vec3):
        """ 
            Solid angle PDF for a specific incident direction - BRDF sampling
            Some PDF has nothing to do with backward incid (from eye to the surface), like diffusive 
            This PDF is actually the PDF of cosine-weighted term * BRDF function value
        """
        pdf = 0.0
        dot_outdir = tm.dot(it.n_s, outdir)
        dot_indir  = tm.dot(it.n_s, incid)
        if dot_outdir * dot_indir < 0.:         # same hemisphere         
            if self._type == BRDFTag.BLING_PHONG:
                pdf = dot_outdir * INV_PI       # dot is cosine term
            elif self._type == BRDFTag.LAMBERTIAN or self._type == BRDFTag.OREN_NAYAR:
                pdf = dot_outdir * INV_PI
            elif self._type == BRDFTag.MOD_PHONG:
                glossiness      = self.mean[2]
                reflect_view, _ = inci_reflect_dir(incid, it.n_s)
                dot_ref_out     = tm.max(0., tm.dot(reflect_view, outdir))
                diffuse_pdf     = dot_outdir * INV_PI
                specular_pdf    = 0.5 * (glossiness + 1.) * INV_PI * tm.pow(dot_ref_out, glossiness)
                diffuse_color   = ti.select(it.is_tex_invalid(), self.k_d, it.tex)
                pdf = diffuse_color.max() * diffuse_pdf + self.k_s.max() * specular_pdf
            elif self._type == BRDFTag.THIN_COAT:
                reflect, _ = inci_reflect_dir(incid, it.n_s)
                in_ref_F = self.thin_coat_fresnel(it, incid)
                pdf = ti.select(ti.abs(tm.dot(outdir, reflect)) > (1. - 1e-3), in_ref_F, (1. - in_ref_F) * dot_outdir * INV_PI)
            elif self._type == BRDFTag.FRESNEL_BLEND:
                half_vec = (outdir - incid).normalized()
                dot_half = tm.dot(half_vec, it.n_s)
                R = rotation_between(vec3([0, 1, 0]), it.n_s)
                cos_phi2, sin_phi2 = self.fresnel_cos2_sin2(half_vec, it.n_s, R, dot_half)
                pdf = self.k_g[2] * tm.pow(dot_half, self.k_g[0] * cos_phi2 + self.k_g[1] * sin_phi2) / ti.abs(tm.dot(incid, half_vec))
                pdf = 0.5 * (pdf + dot_outdir * INV_PI)
            else:
                if ti.static(__ENABLE_MICROFACET__):
                    if self._type == BRDFTag.MICROFACET:
                        wh = (outdir - incid).normalized()
                        pdf = trow_reitz_pdf(-incid, wh, self.k_g, it.n_s) / (-4. * tm.dot(wh, incid))
        return pdf
    