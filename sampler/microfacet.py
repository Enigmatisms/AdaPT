"""
    Probabilistic model for microfacet surface reflection / transmission model
    This model basically follows the model from PBR-book
    reference: https://www.pbr-book.org/3ed-2018/Reflection_Models/Microfacet_Models
    @author: Qianyue He
    @date: 2023-07-19

    Here, Trowbridge-Reitz (GGX) model is implemented, since it has a longer tail effect
    (should look more different than Beckmann)
"""

import taichi as ti
import taichi.math as tm
from taichi.math import vec3, vec4
from la.cam_transform import localize_rotate
from renderer.constants import PI2, PI_DIV2

__all__ = ["trow_reitz_D", "trow_reitz_G", "trow_reitz_sample_wh", "trow_reitz_pdf", "convert_to_raw"]

__EPS__ = 1e-5

@ti.dataclass
class SampledValues:
    """ For Trowbridge-Reitz sampled values """
    slope_x: float
    slope_y: float

@ti.func
def trow_reitz_D(raw_vec: vec4, alphas: vec3):
    """ Trowbridge-Reitz PDF 

        raw_vec: if the half vector is sampled (which usually is) then sampling directly computes
            cos_theta (w.r.t shading normal), sin_theta and phi, then we will be saved from coverting
            half vector back to these values
            raw_vec: [cos_theta, sin_theta, cos_phi, sin_phi]
    """
    pdf = 0
    if raw_vec[0] > 0:
        wh_dot2 = raw_vec[0] * raw_vec[0]
        wh_dot4 = wh_dot2 * wh_dot2
        tan_theta2 = raw_vec[1] * raw_vec[1] / wh_dot2
        alpha_x, alpha_y, _ = alphas
        pdf = ti.exp(-tan_theta2 * (raw_vec[2] / (alpha_x * alpha_x) +
                                  raw_vec[3] / (alpha_y * alpha_y))) / \
           (ti.pi * alpha_x * alpha_y * wh_dot4)
    return pdf

@ti.func
def convert_to_raw(d_in: vec3, normal: vec3) -> vec4:
    """ Sometimes... we won't have a raw_vec, therefore we need to produce it
        get cos_theta / sin_theta / cos_phi / sin_phi via d_in w.r.t to the normal
        note that sin_theta lies in [0, 1], cos_theta, however, is in [-1, 1]
    """
    # vec3([tm.cos(phi) * sin_theta, cos_theta, tm.sin(phi) * sin_theta])
    local_dir = localize_rotate(normal, d_in)
    cos_theta = local_dir[1]
    sin_theta = ti.sqrt(ti.max(0., 1. - cos_theta * cos_theta))
    cos_phi = 1.
    sin_phi = 0.
    if sin_theta > __EPS__:
        cos_phi = local_dir[0] / sin_theta
        sin_phi = local_dir[2] / sin_theta
    return vec4([cos_theta, sin_theta, cos_phi, sin_phi])

@ti.func
def trow_reitz_lambda(raw_vec: vec4, alphas: vec3):
    """ Lambda for Trowbridge-Reitz """
    value = 0
    abs_cos_theta = ti.abs(raw_vec[0])
    if abs_cos_theta > __EPS__:
        abs_tan_theta = raw_vec[1] / abs_cos_theta
        alpha_x, alpha_y, _ = alphas
        alpha = ti.sqrt(raw_vec[2] * raw_vec[2] * alpha_x * alpha_x + raw_vec[3] * raw_vec[3] * alpha_y * alpha_y)
        alpha_tan2 = alpha * abs_tan_theta
        alpha_tan2 *= alpha_tan2
        value = (-1. + ti.sqrt(1. + alpha_tan2)) * 0.5
    return value

@ti.experimental.real_func
def __trow_reitz_sample(cos_theta: float) -> SampledValues:
    u1 = ti.random(float)
    u2 = ti.random(float)
    if cos_theta > 1 - __EPS__:
        r = ti.sqrt(u1 / (1. - u1))
        phi = 6.28318530718 * u2
        return SampledValues(slope_x = r * ti.cos(phi), slope_y = r * ti.sin(phi))

    sin_theta = ti.sqrt(ti.max(0., 1 - cos_theta * cos_theta))
    tan_theta = sin_theta / cos_theta
    G1 = 2. / (1. + ti.sqrt(1. + tan_theta * tan_theta))

    A = 2. * u1 / G1 - 1.
    tmp = ti.min(1e10, 1. / (A * A - 1.))
    D = ti.sqrt(
        ti.max((tan_theta * tan_theta * tmp * tmp - (A * A - tan_theta * tan_theta) * tmp), 0.))
    slope_x_1 = tan_theta * tmp - D
    slope_x_2 = slope_x_1 + D * 2.
    slope_x = ti.select((A < 0) or (slope_x_2 > 1. / tan_theta), slope_x_1, slope_x_2)

    S = 1.
    if u2 > 0.5:
        S = 1.
        u2 = 2.0 * (u2 - 0.5)
    else:
        S = -1.
        u2 = 2. * (0.5 - u2)
    z = (u2 * (u2 * (u2 * 0.27385 - 0.73369) + 0.46341)) / \
        (u2 * (u2 * (u2 * 0.093073 + 0.309420) - 1.0) + 0.597999)
    slope_y = S * z * ti.sqrt(1. + slope_x * slope_x)
    return SampledValues(slope_x = slope_x, slope_y = slope_y)

@ti.func
def trow_reitz_sample(incid: vec3, normal: vec3, alpha_x: float, alpha_y: float):
    """ This is the callable Trowbridge-Reitz sampling function
        this normal vector is 
        TODO: please be careful about the incident direction (should be pointing outwards)
    """
    coeff = vec3([alpha_x, alpha_y, 1])
    stretch_incid = (incid * coeff).normalized()

    cos_theta, _, cos_phi, sin_phi = convert_to_raw(stretch_incid, normal)
    sampled_slopes = __trow_reitz_sample(cos_theta)
    slope_x = sampled_slopes.slope_x
    slope_y = sampled_slopes.slope_y

    tmp     = cos_phi * slope_x - sin_phi * slope_y
    slope_y = sin_phi * slope_x + cos_phi * slope_y
    slope_x = tmp

    slope_x = alpha_x * slope_x
    slope_y = alpha_y * slope_y

    # This should be correct for our local coorinate system [0, 1, 0]
    return vec3([-slope_x, 1., -slope_y]).normalized()

@ti.func
def trow_reitz_G1(direct: vec3, alphas: vec3):
    return 1. / (1. + trow_reitz_lambda(direct, alphas))

@ti.func
def trow_reitz_G(incid: vec3, outdir: vec3, alphas: vec3):
    """ TODO: check the direction for the incid/outdir, which should point outwards """
    return 1. / (1. + trow_reitz_lambda(incid, alphas) + trow_reitz_lambda(outdir, alphas))

@ti.func
def trow_reitz_sample_wh_whole(incid: vec3, alpha_x: float, alpha_y: float):
    """ This function might not be used (only useful when we sample outside of the visible area)
        We follow a backward path trace convention, but to be more 'intuitive'
        incid is actually the ray (starting from camera) direction (for UDPT)
    """
    cos_theta = 0
    u1 = ti.random(float)
    u2 = ti.random(float)
    phi = PI2 * u2
    if alpha_x == alpha_y:
        tan_theta2 = alpha_x * alpha_x * u1 / (1. - u1)
        cos_theta = 1. / ti.sqrt(1 + tan_theta2)
    else:
        phi = tm.atan2(alpha_y * ti.tan(PI2 * u2 + PI_DIV2), alpha_x)
        if u2 > 0.5:
            phi += ti.pi
        alpha_x2 = alpha_x * alpha_x
        alpha_y2 = alpha_y * alpha_y
        alpha2 = 1. / (cos_phi * cos_phi / alpha_x2 + sin_phi * sin_phi / alpha_y2)
        tan_theta2 = alpha2 * u1 / (1 - u1)
        cos_theta = 1 / ti.sqrt(1 + tan_theta2)
    sin_phi = tm.sin(phi)
    cos_phi = tm.cos(phi)
    sin_theta = ti.sqrt(ti.max(0., 1. - cos_theta * cos_theta))
    wh = vec3([cos_phi * sin_theta, cos_theta, sin_phi * sin_theta])
    raw_vec = vec4([cos_theta, sin_theta, cos_phi, sin_phi])
    if tm.dot(wh, incid) < 0:
        raw_vec = vec4([-cos_theta, sin_theta, -cos_phi, -sin_phi])
        wh = -wh
    return wh, raw_vec

@ti.func
def trow_reitz_sample_wh(incid: vec3, normal: vec3, alpha_x: float, alpha_y: float):
    dot_incid = tm.dot(incid, normal)
    flip = dot_incid > 0
    wh = trow_reitz_sample(ti.select(flip, incid, -incid), alpha_x, alpha_y)
    if flip:
        wh = -wh
    return wh

@ti.func
def trow_reitz_pdf(incid: vec3, wh: vec3, normal: vec3):
    """ wo in the pbr-book actually means incident ray (ray from camera) """
    return trow_reitz_D(wh) * trow_reitz_G1(incid) * ti.abs(tm.dot(wh, incid)) / \
        ti.abs(tm.dot(normal, incid))
