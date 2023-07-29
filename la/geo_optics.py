"""
    Some geometric optics functions
    @author: Qianyue He
    @date: 2023-2-4
"""

import taichi as ti
import taichi.math as tm
from taichi.math import vec3

__all__ = ['inci_reflect_dir', 'exit_reflect_dir', 'schlick_fresnel', 
           'fresnel_equation', 'snell_refraction', 'is_total_reflection', 'fresnel_eval']

@ti.func
def inci_reflect_dir(ray: vec3, normal: vec3):
    dot = tm.dot(normal, ray)
    return (ray - 2 * normal * dot).normalized(), dot

@ti.func
def exit_reflect_dir(ray: vec3, normal: vec3):
    dot = tm.dot(normal, ray)
    return (2 * normal * dot - ray).normalized(), dot

@ti.func
def schlick_fresnel(r_s: vec3, dot_val: float):
    """ Schlick's Fresnel Fraction Approximation [1993] """
    return r_s + (1 - r_s) * tm.pow(1. - dot_val, 5)

@ti.experimental.real_func
def fresnel_eval(cos_v: float, n_in: float, n_tr: float) -> float:
    """ Evaluate Fresnel Equation with only one cosine value input 
        n_in: incident (outside) medium IOR
        n_tr: transmission (inside) medium IOR
    """
    neg_cos_v = cos_v < 0
    # if the ray points outwards
    cos_value = ti.select(neg_cos_v, -cos_v, cos_v)
    ior_in = ti.select(neg_cos_v, n_tr, n_in)
    ior_tr = ti.select(neg_cos_v, n_in, n_tr)
    # refraction cosine
    sin_v = ti.sqrt(ti.max(0, 1. - cos_value * cos_value))
    sin_t = ior_in / ior_tr * sin_v
    cos_tr = ti.sqrt(ti.max(0, 1. - sin_t * sin_t))
    return fresnel_equation(ior_in, ior_tr, cos_value, cos_tr)

@ti.func
def fresnel_equation(n_in: float, n_out: float, cos_inc: float, cos_ref: float):
    """ 
        Fresnel Equation for calculating specular ratio
        Since Schlick's Approximation is not clear about n1->n2, n2->n1 (different) effects

        This Fresnel equation is for dielectric, not for conductor
    """
    n1cos_i = n_in * cos_inc
    n2cos_i = n_out * cos_inc
    n1cos_r = n_in * cos_ref
    n2cos_r = n_out * cos_ref
    rs = (n1cos_i - n2cos_r) / (n1cos_i + n2cos_r)
    rp = (n1cos_r - n2cos_i) / (n1cos_r + n2cos_i)
    return 0.5 * (rs * rs + rp * rp)

@ti.func
def is_total_reflection(dot_normal: float, ni: float, nr: float):
    return (1. - ti.pow(ni / nr, 2) * (1. - ti.pow(dot_normal, 2))) < 0.

@ti.func
def snell_refraction(incid: vec3, normal: vec3, dot_n: float, ni: float, nr: float):
    """ Refraction vector by Snell's Law, note that an extra flag will be returned """
    exiting    = tm.sign(dot_n)
    ratio      = ni / nr
    cos_r2     = 1. - ti.pow(ratio, 2) * (1. - ti.pow(dot_n, 2))
    valid      = False              # for ni > nr situation, there will be total reflection
    refra_vec  = vec3([0, 0, 0])
    if cos_r2 > 0.:                 # cos_r2 > 0. always holds if ni < nr
        valid = True
        refra_vec = (ratio * incid - ratio * dot_n * normal + exiting * ti.sqrt(cos_r2) * normal).normalized()
    return refra_vec, valid
