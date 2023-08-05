import taichi as ti
import taichi.math as tm

from misc import *
from la.cam_transform import convert_to_raw
from la.geo_optics import snell_refraction, fresnel_equation
from renderer.constants import INV_PI

# ======================= Oren-Nayar (PBR Rough) =============================
@ti.experimental.real_func
def eval_oren_nayar(ipt: OrenNayarInput) -> OrenNayarOutput:
    """ Note that Oren-Nayar is physically based diffuse material 
        Therefore, we can just sample according to the cosine-hemispherical distribution
        Then, we only need to implement evaluating function
    """
    raw_wi = convert_to_raw(-ipt.ray_in, ipt.n_s)
    raw_wo = convert_to_raw(ipt.ray_out, ipt.n_s)

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
    diffuse_color = ti.select(ipt.is_tex_invalid(), ipt.k_d, ipt.tex)
    # ti.abs(raw_wo[0]) is cosine term
    result = diffuse_color * INV_PI * (ipt.k_g[0] + ipt.k_g[1] * max_cos * sin_alpha * tan_beta) * ti.abs(raw_wo[0])
    return OrenNayarOutput(f_val = result)
# ============================================================================

# =================== Thin coat (Fresnel coating for plastic material) ================
@ti.experimental.real_func
def thin_coat_fresnel(ipt: OrenNayarInput) -> float:
    """ Evaluating the Fresnel coating: mirror specular + Oren-Nayar diffuse 
        note that evaluation for mirror specular part is always 0

        in_ref_F can be precomputed (during sampling)
    """
    # input Fresnel evaluation
    dot_in = tm.dot(ipt.ray_in, ipt.n_s)
    # We will not have total reflection for incident ray
    refra_in, _v = snell_refraction(ipt.ray_in, ipt.n_s, dot_in, 1.0, ipt.k_g[0])
    in_ref_F = fresnel_equation(1., ipt.k_g[0], ti.abs(dot_in), ti.abs(tm.dot(refra_in, ipt.n_s)))

    # output Fresnel evaluation
    dot_out = tm.dot(ipt.ray_out, ipt.n_s)
    refra_out, _v = snell_refraction(ipt.ray_out, ipt.n_s, dot_out, ipt.k_g[0], 1.0)    # always valid
    out_ref_F = fresnel_equation(ipt.k_g[0], 1.0, ti.abs(dot_out), ti.abs(tm.dot(refra_out, ipt.n_s)))
    
    return (1. - out_ref_F) * (1. - in_ref_F)
# ============================================================================