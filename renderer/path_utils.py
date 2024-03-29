"""
    Vertex definition and some other utility function for BDPT
    @author: Qianyue He
    @date: 2023-2-23
"""

import taichi as ti
import taichi.math as tm
from taichi.math import vec3
from renderer.constants import *
from tracer.interaction import Interaction

@ti.func
def remap_pdf(x: float) -> float:
    return ti.select(x > 0., x, 1.)

@ti.dataclass
class Vertex:
    """
        A (64 + 24)-Byte Path Vertex
    """
    _type:      ti.i8
    """ 0 for surface, 1 for medium, 2 for light, 3 for camera """
    obj_id:     ti.i8       
    """ hit object (BSDF) id """
    emit_id:    ti.i8       
    """ if the vertex is on a area emitter, just store the emitter info """
    bool_bits:  ti.i8
    """ Bool bits: [0 pos delta, 1 dir delta, 2 is area, 3 is inifite, 4 is in free space, 5 is specular delta, others reserved] """
    pdf_fwd:    float      
    """ forward pdf """
    pdf_bwd:    float      
    """ backward pdf """
    time:       float      
    """ hit time (reserved for future uses) """
    n_s:     vec3        
    """ hit normal (shading normal) """
    n_g:        vec3
    """ hit normal (geometric normal) """
    pos:        vec3        
    """ hit pos """
    ray_in:     vec3        
    """ incident ray direction (towards pos) """
    beta:       vec3        
    """ path throughput """
    tex:        vec3        
    """ texture color (used to calculate color during BDPT connection) """

    @ti.func
    def set_pdf_bwd(self, pdf: float, next_point: vec3):
        self.pdf_bwd = self.get_pdf_bwd(pdf, next_point)

    @ti.func
    def get_pdf_bwd(self, pdf: float, next_point: vec3):
        if pdf > 0:
            diff_vec = self.pos - next_point
            inv_norm2 = 1. / diff_vec.norm_sqr()
            pdf *= inv_norm2
            if self.has_normal():      # camera has no normal, for now (pin-hole)
                pdf *= ti.abs(tm.dot(self.n_s, diff_vec * ti.sqrt(inv_norm2)))
        return pdf

    @ti.func
    def is_connectible(self):
        connectible = True
        if self._type == VERTEX_SURFACE or self._type == VERTEX_EMITTER:
            # TODO: check whether all vertices are connectible (against PBRT-v3) -- for surface vertices
            connectible = (self.bool_bits & 0x02) == 0          # not directional delta
        return connectible
    
    @ti.func
    def is_in_free_space(self):
        return self.bool_bits & 0x10
    
    @ti.func
    def has_normal(self):
        # Point Source / Area Source / Surface interaction vertex all have a normal
        return (self.n_s != ZERO_V3).any()
    
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
    
    @ti.func
    def not_delta_source(self):
        # if an emitter is positional delta or directional delta
        return (self.bool_bits & 0x03) == 0
    
    @ti.func
    def get_interaction(self) -> Interaction:
        return Interaction(obj_id = self.obj_id, n_s = self.n_s, n_g = self.n_g, tex = self.tex)
    
    @ti.func
    def set_d_delta(self):
        self.bool_bits |= ti.cast(0x22, ti.i8)
    