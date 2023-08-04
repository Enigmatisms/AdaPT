"""
    General sampling functions for direction sampling
    sampling azimuth angle and zenith angle
    @author: Qianyue He
    @date: 2023-1-27
"""

import taichi as ti
import taichi.math as tm
from taichi.math import vec3
from renderer.constants import INV_PI, INV_2PI, PI2, ZERO_V3, PI_DIV4, PI_DIV2

__all__ = ['cosine_hemisphere', 'uniform_hemisphere', 'sample_triangle', 
            'balance_heuristic', 'mod_phong_hemisphere', 'fresnel_hemisphere', 'random_rgb']

@ti.func
def random_rgb(vector):
    """ choose one spectrum (RGB) component randomly """
    idx = ti.random(int) % 3
    return ti.max(vector[idx], 1e-5)

@ti.func
def cosine_hemisphere():
    """
        Zenith angle (cos theta) follows a ramped PDF (triangle like)
        Azimuth angle (itself) follows a uniform distribution
    """
    eps = ti.random(float)
    cos_theta = ti.sqrt(eps)       # zenith angle
    sin_theta = ti.sqrt(1. - eps)
    phi = PI2 * ti.random(float)         # uniform dist azimuth angle
    pdf = cos_theta * INV_PI        # easy to deduct, just try it
    # rotational offset w.r.t axis [0, 1, 0] & pdf
    
    return vec3([tm.cos(phi) * sin_theta, cos_theta, tm.sin(phi) * sin_theta]), pdf

@ti.func
def mod_phong_hemisphere(alpha: float):
    """ 
        PDF for modified Phong model 
        Lafortune & Willems, Using the Modified Phong Reflectance Model for Physically Based Rendering, 1994
    """
    cos_theta = tm.pow(ti.random(float), 1. / (alpha + 1.))
    sin_theta = ti.sqrt(1. - cos_theta * cos_theta)
    phi = PI2 * ti.random(float) 
    pdf = 0.5 * (1. + alpha) * tm.pow(cos_theta, alpha) * INV_PI
    return vec3([tm.cos(phi) * sin_theta, cos_theta, tm.sin(phi) * sin_theta]), pdf

@ti.func
def uniform_hemisphere():
    """ Both zenith (cosine) and azimuth angle (original) are uniformly distributed """
    cos_theta = ti.random(float)
    sin_theta =  ti.sqrt(1 - cos_theta * cos_theta)
    phi = PI2 * ti.random(float)
    return vec3([tm.cos(phi) * sin_theta, cos_theta, tm.sin(phi) * sin_theta]), INV_2PI

@ti.func
def uniform_sphere():
    """ Uniform direction sampling on a sphere """
    cos_theta = 2. * ti.random(float) - 1.
    sin_theta =  ti.sqrt(1 - cos_theta * cos_theta)
    phi = PI2 * ti.random(float)
    return vec3([tm.cos(phi) * sin_theta, cos_theta, tm.sin(phi) * sin_theta]), INV_2PI * 0.5

@ti.func
def uniform_cone(cos_range = 1.0):
    """ Uniform direction sampling on a sphere """
    epsilon = ti.random(float)
    cos_theta = 1. - epsilon + cos_range * epsilon
    sin_theta =  ti.sqrt(1 - cos_theta * cos_theta)
    phi = PI2 * ti.random(float)
    return vec3([tm.cos(phi) * sin_theta, cos_theta, tm.sin(phi) * sin_theta])

@ti.func
def concentric_disk_sample():
    off_x = ti.random(float) * 2. - 1.
    off_y = ti.random(float) * 2. - 1.
    result = ZERO_V3
    if off_x != 0 and off_y != 0:
        if ti.abs(off_x) > ti.abs(off_y):
            theta = PI_DIV4 * (off_y / off_x)
            result = vec3([off_x * ti.cos(theta), 0., off_x * ti.sin(theta)])
        else:
            theta = PI_DIV2 - PI_DIV4 * (off_x / off_y)
            result = vec3([off_y * ti.cos(theta), 0., off_y * ti.sin(theta)])
    return result

@ti.func
def fresnel_hemisphere(nu: float, nv: float):
    eps1 = ti.random(float) * 4.
    inner_angle = eps1 - tm.floor(eps1)
    tan_phi = ti.sqrt((nu + 1) / (nv + 1)) * ti.tan(tm.pi / 2 * inner_angle)
    cos_phi2 = 1. / (1. + tan_phi ** 2)
    sin_phi2 = 1. - cos_phi2
    cos_phi = ti.sqrt(cos_phi2)
    if eps1 > 1. and eps1 <= 3.: cos_phi *= -1.
    sin_phi = ti.sqrt(sin_phi2) * tm.sign(2. - eps1)
    power_coeff = nu * cos_phi2 + nv * sin_phi2
    cos_theta = tm.pow(1. - ti.random(float), 1. / (power_coeff + 1.))
    sin_theta = ti.sqrt(1 - cos_theta * cos_theta)
    return vec3([cos_phi * sin_theta, cos_theta, sin_phi * sin_theta]), power_coeff

@ti.func
def sample_triangle(dv1: vec3, dv2: vec3):
    """ Sample on a mesh triangle """
    u1 = ti.random(float)
    u2 = ti.random(float)
    triangle_pt = dv1 * u1 + dv2 * u2
    if u1 + u2 > 1.0:
        triangle_pt = dv1 + dv2 - triangle_pt
    return triangle_pt

@ti.func
def balance_heuristic(pdf_a: float, pdf_b: float):
    """ Balanced heuristic function for MIS weight computation """
    return ti.select(pdf_a > 1e-7, pdf_a / (pdf_a + pdf_b), 0.)
    