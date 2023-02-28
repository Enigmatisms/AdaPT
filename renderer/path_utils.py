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
    bool_bits:  ti.i8       # bit value: [0: is_delta, 1: in_free_space, 2-8: reserved]

    pdf_fwd:    float      # forward pdf
    pdf_bwd:    float      # backward pdf
    time:       float      # hit time (reserved for future uses)

    normal:     vec3        # hit normal
    pos:        vec3        # hit pos
    ray_in:     vec3        # incident ray direction (towards pos)
    beta:       vec3        # path throughput

    @ti.func
    def set_pdf_bwd(self, pdf: float, prev_point: vec3):
        diff_vec = self.pos - prev_point
        inv_norm2 = 1. / diff_vec.norm_sqr()
        pdf *= inv_norm2
        if self._type == 0 or self._type == 2:      # camera has no normal, for now (pin-hole)
            pdf *= ti.abs(tm.dot(self.normal, diff_vec * ti.sqrt(inv_norm2)))
        self.pdf_bwd = pdf

    @ti.func
    def convert_density(self, next_v: ti.template(), pdf, ray):
        """ Vertex method for converting solid angle density to unit area measure """
        depth = ray.norm()
        ray  /= depth
        if next_v.on_surface():
            pdf *= ti.abs(tm.dot(next_v.normal, ray))
        return pdf / (depth * depth)

    @ti.func
    def get_pdf_context(self):
        """ FIXME: save this for future uses, if `pdf` passes correctly, this will be deprecated """
        return int(self.obj_id), self.ray_in, self.normal, self._type == VERTEX_MEDIUM, self.is_in_free_space(), self.pos
        
    @ti.func
    def is_connectible(self):
        # the [0]th bit of bool bits indicates whether the vertex is delta, if NOT delta then it's connectible
        return (self.bool_bits & 0x01) == 0
        
    @ti.func
    def is_in_free_space(self):
        return self.bool_bits & 0x02
    
    @ti.func
    def on_surface(self):
        # The vertex on the surface or vertex is on an [area] emitter
        return (self._type == VERTEX_SURFACE) or (self._type == VERTEX_EMITTER and self.is_connectible())
    
    @ti.func
    def pdf_ratio(self):
        return remap_pdf(self.pdf_bwd) / remap_pdf(self.pdf_fwd)
    
    @ti.func
    def is_mi(self):
        return self._type == VERTEX_MEDIUM
    