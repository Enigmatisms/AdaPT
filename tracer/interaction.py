"""
    We really need a interaction class to manage 
    the ever-increasing complexity introduced by ray intersection
    @author Qianyue He
    @date 2023.6.18
"""

import taichi as ti
from taichi.math import vec2, vec3

@ti.dataclass
class Interaction:
    """ Surface interaction management class """

    obj_id:     int
    """ index for object intersected """
    prim_id:    int
    """ primitive for object intersected """
    n_s:        vec3
    """ shading normal """
    n_g:        vec3
    """ geometric normal """
    tex:        vec3
    """ Texture color. Maybe we need more fields like roughness, etc. """   
    uv:         vec2
    """ uv_coordinates. When returing from ray_intersect, this is local uv
        Local uv should be post-processed to be global uv
    """
    min_depth:  float
    """ minimum depth to a surface """

    # obj_id, normal, min_depth, u_coord, v_coord

    @ti.func
    def is_ray_not_hit(self):
        return self.obj_id < 0
    
    @ti.func
    def is_tex_invalid(self):
        return self.tex[0] < 0