""" Miscellaneous utilities
    @author Qianyue He
    @date 2023.8.4
"""

import taichi as ti
from taichi.math import vec3

__all__ = ["OrenNayarInput", "OrenNayarOutput"]

@ti.dataclass
class OrenNayarInput:
    """ To reduce compile-time overhead (due to inlining)           \\
        and consider that Oren-Nayar is widely used, we convert     \\
        the evaluation function into a ti.real_func, therefore      \\
        the input and the output should be wrapped in structs
    """

    ray_in: vec3
    """ incident ray direction (pointing inwards normally) """

    ray_out: vec3
    """ exiting ray direction (pointing outwards normally) """

    n_s: vec3
    """ Shading normal """

    tex: vec3
    """ Texture record """

    k_d: vec3
    """ diffuse parameters """

    k_g: vec3
    """ glossy parameters """

    @ti.func
    def is_tex_invalid(self):
        return self.tex[0] < 0

@ti.dataclass
class OrenNayarOutput:
    f_val: vec3
    """ Evaluated BRDF spectrum value """
