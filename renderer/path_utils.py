"""
    Vertex definition and some other utility function for BDPT
    @author: Qianyue He
    @date: 2023-2-23
"""

import taichi as ti
import taichi.math as tm
from taichi.math import vec3
from renderer.constants import *

@ti.func
def remap_pdf(x: float) -> float:
    return ti.select(x > 0., x, 1.)

@ti.dataclass
class Vertex:
    """
        A 64-Byte Path Vertex
    """
    _type:      ti.i8       # 0 for surface, 1 for medium, 2 for light, 3 for camera
    obj_id:     ti.i8       # hit object (BSDF) id
    emit_id:    ti.i8       # if the vertex is on a area emitter, just store the emitter info

    # Bool bits: [0 pos delta, 1 dir delta, 2 is area, 3 is inifite, 4 is in free space, 5 is specular delta, others reserved]
    bool_bits:  ti.i8

    pdf_fwd:    float      # forward pdf
    pdf_bwd:    float      # backward pdf
    time:       float      # hit time (reserved for future uses)

    normal:     vec3        # hit normal
    pos:        vec3        # hit pos
    ray_in:     vec3        # incident ray direction (towards pos)
    beta:       vec3        # path throughput

    @ti.func
    def set_pdf_bwd(self, pdf: float, next_point: vec3):
        diff_vec = self.pos - next_point
        inv_norm2 = 1. / diff_vec.norm_sqr()
        pdf *= inv_norm2
        if self.has_normal():      # camera has no normal, for now (pin-hole)
            pdf *= ti.abs(tm.dot(self.normal, diff_vec * ti.sqrt(inv_norm2)))
        self.pdf_bwd = pdf

    @ti.func
    def is_connectible(self):
        connectible = True
        if self._type == VERTEX_SURFACE or self._type == VERTEX_EMITTER:
            connectible = (self.bool_bits & 0x02) == 0          # not directional delta
        return connectible
    
    @ti.func
    def is_in_free_space(self):
        return self.bool_bits & 0x10
    
    @ti.func
    def has_normal(self):
        # Point Source / Area Source / Surface interaction vertex all have a normal
        return (self.normal != ZERO_V3).any()
    
    @ti.func
    def pdf_ratio(self):
        return remap_pdf(self.pdf_bwd) / remap_pdf(self.pdf_fwd)
    
    @ti.func
    def is_mi(self):
        return self._type == VERTEX_MEDIUM
    
    @ti.func
    def is_light(self):
        return (self._type == VERTEX_EMITTER) or (self.bool_bits & 0x04)
    
    @ti.func
    def not_delta(self):
        return (self.bool_bits & 0x20) == 0
    