"""
    Phase function definition
    currently, only H-G, multiple-HG (up to 3) and Rayleigh (non-spectral one) is supported
"""

"""
    Volume Scattering Medium
    For one given object, object can have [BSDF + Medium], [BSDF], [BRDF], which corresponds to
    (1) non-opaque object with internal scattering (2) non-opaque object without scattering (3) opaque object
    @author: Qianyue He
    @date: 2023-2-7
"""

import taichi as ti
import taichi.math as tm
from taichi.math import vec3
from sampler.phase_sampling import *
from renderer.constants import INV_2PI

@ti.func
def phase_hg(cos_theta: float, g: float):
    g2 = g * g
    denom = 1. + g2 - 2. * g * cos_theta
    return (1. - g2) / (ti.sqrt(denom) * denom) * 0.5 * INV_2PI

# ============== Rayleigh ================
@ti.func
def phase_rayleigh(cos_theta: float):
    return 0.375 * INV_2PI * (1. + cos_theta * cos_theta)

@ti.dataclass
class PhaseFunction:
    _type:  int
    par:    vec3
    pdf:    vec3            # for multiple H-G, an extra pdf is required
    
    @ti.func
    def sample_p(self, incid: vec3):
        """ 
            Phase function returns a float, it's depended on wavelength though 
            I don't account for wavelength currently, since sources can have multiple wavelengths
            Mie scattering is not implemented currently
        """
        ret_dir = incid
        ret_p   = 1.0
        if self._type == 0:             # H-G
            g = self.par[0]
            ret_dir, cos_t = sample_hg(g)
            ret_p = phase_hg(cos_t, g)
        elif self._type == 1:           # multiple H-G
            eps = ti.random(float)
            cos_t = 0.
            g = 0.
            if eps < self.pdf[0]:
                g = self.par[0]
            elif eps < self.pdf[0] + self.pdf[1]:
                g = self.par[1]
            else:
                g = self.par[2]
            ret_dir, cos_t = sample_hg(g)
            ret_p = phase_hg(cos_t, g)
        elif self._type == 2:           # Rayleigh
            ret_dir, cos_t = sample_rayleigh()
            ret_p = phase_rayleigh(cos_t)
        return ret_dir, ret_p
    
    @ti.func
    def eval_p(self, ray_in: vec3, ray_out: vec3):
        """ For phase functions, output value is PDF, therefore PDF function is not needed. """
        ret_p   = 1.0
        cos_theta = -tm.dot(ray_in, ray_out)        # ray_in points to x, ray_out points away from x
        if self._type == 0:             # H-G: FIXME: single H-G has no spectral differences
            # TODO: the spectral difference is actually simple, we can do this according to mfp sampling
            ret_p = phase_hg(cos_theta, self.par[0])
        elif self._type == 1:           # multiple H-G
            ret_p = phase_hg(cos_theta, self.par[0]) * self.pdf[0] + phase_hg(cos_theta, self.par[1]) * self.pdf[1]
            if self.pdf[1] > 1e-4:
                ret_p += phase_hg(cos_theta, self.par[2]) * self.pdf[2]
        elif self._type == 2:           # Rayleigh
            ret_p = phase_rayleigh(cos_theta)
        return ret_p
    
    # ============== Specific implementation ================
    # ============== H-G & multiple H-G ================
    