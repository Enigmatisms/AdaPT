"""
    General sampling functions for direction sampling
    sampling azimuth angle and zenith angle
    @author: Qianyue He
    @date: 2023-1-27
"""

import taichi as ti
import taichi.math as tm
from taichi.math import vec3

__all__ = ['cosine_hemisphere', 'uniform_hemisphere', 'sample_triangle', 
            'balance_heuristic', 'mod_phong_hemisphere', 'frensel_hemisphere', 'random_rgb']

pi_inv = 1. / tm.pi

@ti.func
def random_rgb(vector):
    """ Taichi does not support dynamic indexing (it does actually, yet I don't want to set `dynamic_index = True`)"""
    idx = ti.random(int) % 3
    result = 1.0
    if idx == 0:
        result = vector[0]
    elif idx == 1:
        result = vector[1]
    else:
        result = vector[2]
    return ti.max(result, 1e-5)

@ti.func
def cosine_hemisphere():
    """
        Zenith angle (cos theta) follows a ramped PDF (triangle like)
        Azimuth angle (itself) follows a uniform distribution
    """
    eps = ti.random(float)
    cos_theta = ti.sqrt(eps)       # zenith angle
    sin_theta = ti.sqrt(1. - eps)
    phi = 2. * tm.pi * ti.random(float)         # uniform dist azimuth angle
    pdf = cos_theta * pi_inv        # easy to deduct, just try it
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
    phi = 2. * tm.pi * ti.random(float) 
    pdf = 0.5 * (1. + alpha) * tm.pow(cos_theta, alpha) * pi_inv
    return vec3([tm.cos(phi) * sin_theta, cos_theta, tm.sin(phi) * sin_theta]), pdf

@ti.func
def uniform_hemisphere():
    """
        Both zenith (cosine) and azimuth angle (original) are uniformly distributed
    """
    cos_theta = ti.random(float)
    sin_theta =  ti.sqrt(1 - cos_theta * cos_theta)
    phi = 2. * tm.pi * ti.random(float)
    pdf = 0.5 * pi_inv
    return vec3([tm.cos(phi) * sin_theta, cos_theta, tm.sin(phi) * sin_theta]), pdf

@ti.func
def frensel_hemisphere(nu: float, nv: float):
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
    """
        Sample on a mesh triangle
    """
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
    