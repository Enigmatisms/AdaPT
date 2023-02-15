"""
    General sampling functions for direction sampling
    sampling azimuth angle and zenith angle
    @author: Qianyue He
    @date: 2023-1-27
"""

import taichi as ti
import taichi.math as tm
from taichi.math import vec3

__all__ = ['sample_hg', 'sample_rayleigh']

pi_inv = 1. / tm.pi

@ti.func
def sample_hg(g: float):
    """ H-G sphere sampling: returns sampled direction and cos_theta """
    cos_theta = 0.
    if ti.abs(g) < 1e-3:
        cos_theta = 1. - 2. * ti.random(float)
    else:
        sqr_term = (1. - g * g) / (1. - g + 2. * g * ti.random(float))
        cos_theta = (1. + g * g - sqr_term * sqr_term) / (2. * g)
    sin_theta = ti.sqrt(ti.max(0., 1. - cos_theta * cos_theta))
    phi = 2. * tm.pi * ti.random(ti.float32)
    # rotational offset w.r.t axis [0, 1, 0] & pdf
    return vec3([tm.cos(phi) * sin_theta, cos_theta, tm.sin(phi) * sin_theta]), cos_theta

@ti.func
def sample_rayleigh():
    """ rayleigh sphere sampling: returns sampled direction and cos_theta """
    # TODO: check the distribution of Rayleigh sampling
    rd = 2. * ti.random(float) - 1.
    u = - tm.pow(2. * rd + ti.sqrt(4. * rd * rd + 1.) , 1. / 3.)
    cos_theta = tm.clamp(u - 1. / u, -1., 1.)
    sin_theta = ti.sqrt(ti.max(0., 1. - cos_theta * cos_theta))
    phi = 2. * tm.pi * ti.random(ti.float32)
    return vec3([tm.cos(phi) * sin_theta, cos_theta, tm.sin(phi) * sin_theta]), cos_theta
    