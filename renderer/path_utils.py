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
    def pdf(self, renderer: ti.template(), next_v: ti.template(), prev_v: ti.template()):
        """ Renderer passed in is a reference to BDPT class
            When connect to a new path, end point bwd pdf should be updated
            PDF is used when all three points are presented, next_v is directly modified (no race condition)
            Note that when calling `pdf`, self can never be VERTEX_CAMERA (you can check the logic)
        """
        if self._type == VERTEX_EMITTER:
            next_v.pdf_bwd = self.pdf_light(renderer, next_v)
        else:
            is_in_fspace = self.is_in_free_space()
            ray_in = self.pos - next_v.pos
            normed_ray_in = ray_in.normalized()
            """ if prev_v = NULL, yet prev vertex in the path actually exists: self.ray_in is the correct direction or is zero vec
                otherwise, prev vertex does not exist therefore provided with the ertex from another path: calculate """
            ray_out = ti.select(prev_v._type == VERTEX_NULL, -self.ray_in, (prev_v.pos - self.pos).normalized())
            pdf_sa = renderer.get_pdf(self.obj_id, normed_ray_in, ray_out, self.normal, self._type == VERTEX_MEDIUM, is_in_fspace)
            # convert to area measure for the next node
            next_v.pdf_bwd = self.convert_density(pdf_sa, next_v)

    @ti.func
    def convert_density(self, next_v: ti.template(), pdf, ray):
        """ Vertex method for converting solid angle density to unit area measure """
        depth = ray.norm()
        ray  /= depth
        if next_v.on_surface():
            pdf *= ti.abs(tm.dot(next_v.normal, ray))
        return pdf / (depth * depth)

    @ti.func
    def pdf_light(self, renderer: ti.template(), prev_v: ti.template()):
        """ Calculate directional density (then convert to area measure) for prev_v.pdf_bwd """
        pdf = 0.
        # FIXME: if there is no logic bug, the boundary check `emit_id > 0` can be removed
        if self.emit_id >= 0:
            ray_dir  = prev_v.pos - self.pos
            inv_len  = 1. / ray_dir.norm()
            ray_dir *= inv_len
            pdf = renderer.src_field[self.emit_id].direction_pdf(ray_dir, self.normal)
            if prev_v.on_surface():
                pdf *= ti.max(-tm.dot(ray_dir, prev_v.normal), 0.)
            pdf *= (inv_len * inv_len)
        else:
            print("Warning: Current v can't be non-emitter for pdf_light to be called")
        return pdf

    @ti.func
    def pdf_light_origin(self, renderer: ti.template()):
        """ Calculate density if the current vertex is an emitter vertex """
        # FIXME: if there is no logic bug, the boundary check `emit_id > 0` can be removed
        if self.emit_id >= 0:
            pdf = renderer.src_field[self.emit_id].area_pdf() / float(self.src_num)     # uniform emitter selection
        else:
            print("Warning: Current v can't be non-emitter for pdf_light_origin to be called")
        return pdf

    @ti.func
    def get_pdf_context(self):
        """ FIXME: save this for future uses, if `pdf` passes correctly, this will be deprecated """
        return self.obj_id, self.ray_in, self.normal, self._type == VERTEX_MEDIUM, self.is_in_free_space(), self.pos
        
    @ti.func
    def is_connectible(self):
        return self.bool_bits & 0x01
        
    @ti.func
    def is_in_free_space(self):
        return self.bool_bits & 0x02
    
    @ti.func
    def on_surface(self):
        return (self._type & ON_SURFACE) == 0
    
    @ti.func
    def pdf_ratio(self):
        return remap_pdf(self.pdf_bwd) / remap_pdf(self.pdf_fwd)
    