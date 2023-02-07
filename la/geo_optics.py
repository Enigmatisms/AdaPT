"""
    Some geometric optics functions
    @author: Qianyue He
    @date: 2023-2-4
"""

import taichi as ti
import taichi.math as tm
from taichi.math import vec3

__all__ = ['inci_reflect_dir', 'exit_reflect_dir', 'schlick_frensel', 
           'frensel_equation', 'snell_refraction', 'is_total_reflection']

@ti.func
def inci_reflect_dir(ray: vec3, normal: vec3):
    dot = tm.dot(normal, ray)
    return (ray - 2 * normal * dot).normalized(), dot

@ti.func
def exit_reflect_dir(ray: vec3, normal: vec3):
    dot = tm.dot(normal, ray)
    return (2 * normal * dot - ray).normalized(), dot

@ti.func
def schlick_frensel(r_s: vec3, dot_val: ti.f32):
    """ Schlick's Frensel Fraction Approximation [1993] """
    return r_s + (1 - r_s) * tm.pow(1. - dot_val, 5)

@ti.func
def frensel_equation(n_in: ti.f32, n_out: ti.f32, cos_inc: ti.f32, cos_ref: ti.f32):
    """ 
        Frensel Equation for calculating specular ratio
        Since Schlick's Approximation is not clear about n1->n2, n2->n1 (different) effects
    """
    n1cos_i = n_in * cos_inc
    n2cos_i = n_out * cos_inc
    n1cos_r = n_in * cos_ref
    n2cos_r = n_out * cos_ref
    rs = (n1cos_i - n2cos_r) / (n1cos_i + n2cos_r)
    rp = (n1cos_r - n2cos_i) / (n1cos_r + n2cos_i)
    return 0.5 * (rs * rs + rp * rp)

@ti.func
def is_total_reflection(dot_normal: ti.f32, ni: ti.f32, nr: ti.f32):
    return (1. - ti.pow(ni / nr, 2) * (1. - ti.pow(dot_normal, 2))) < 0.

@ti.func
def snell_refraction(incid: vec3, normal: vec3, dot_n: ti.f32, ni: ti.f32, nr: ti.f32):
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
