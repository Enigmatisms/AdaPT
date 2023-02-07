"""
    Light source abstraction
    FIXME: sampling is not yet implemented currently (2023.1.20 version)
    @date: 2023.1.20
    @author: Qianyue He
"""

import sys
sys.path.append("..")

import numpy as np
import taichi as ti
import xml.etree.ElementTree as xet
from taichi.math import vec3

from scene.general_parser import rgb_parse
from la.cam_transform import delocalize_rotate
from sampler.general_sampling import sample_triangle, cosine_hemisphere

@ti.dataclass
class TaichiSource:
    """
        This implementation is not elegant, too obese. Of course, there are better ways \\
        For example, for every 'variable field', we can place them to bit-masked: \\
        node = pointer -> bitmasked \\
        node.place(pos); node.place(dirv); node.place(base_1); node.place(base_2); \\
        The following implementation is much simpler, and light source will not consume too much memory
    """
    _type:      ti.i32      # 0 Point, 1 Area, 2 Spot, 3 Directional
    obj_ref_id: ti.i32      # Referring to the attaching object
    intensity:  vec3
    pos:        vec3
    dirv:       vec3
    base_1:     vec3
    base_2:     vec3
    l1:         ti.f32
    l2:         ti.f32
    inv_area:   ti.f32      # inverse area (for non-point emitters, like rect-area or mesh attached emitters)

    @ti.func
    def distance_attenuate(self, x: vec3):
        return ti.min(1.0 / ti.max(x.norm_sqr(), 1e-5), 1.0)

    @ti.func
    def sample(
        self, dvs: ti.template(), normals: ti.template(), 
        mesh_cnt: ti.template(), hit_pos: vec3
    ):
        """
            A unified sampling function, choose sampling strategy according to _type \\
            input ray hit point \\
            returns <sampled source point> <souce intensity> <sample pdf> 
        """
        ret_int = self.intensity
        ret_pos = self.pos
        ret_pdf = 1.0
        if self._type == 0:     # point source
            ret_int *= self.distance_attenuate(hit_pos - ret_pos)
        elif self._type == 1:   # area source
            ret_pdf     = self.inv_area
            dot_light   = 1.0
            diff        = vec3([0, 0, 0])
            if self.obj_ref_id >= 0:   # sample from mesh
                mesh_num = mesh_cnt[self.obj_ref_id]
                normal   = vec3([0, 1, 0])
                if mesh_num:
                    tri_id    = ti.random(ti.i32) % mesh_num       # ASSUME that triangles are similar in terms of area
                    normal    = normals[self.obj_ref_id, tri_id]
                    dv1       = dvs[self.obj_ref_id, tri_id, 0]
                    dv2       = dvs[self.obj_ref_id, tri_id, 1]
                    ret_pos   = sample_triangle(dv1, dv2) + dvs[self.obj_ref_id, tri_id, 2]
                else:
                    center    = dvs[self.obj_ref_id, 0, 0]
                    radius    = dvs[self.obj_ref_id, 0, 1][0]
                    to_hit    = (hit_pos - center).normalized()
                    local_dir, pdf = cosine_hemisphere()
                    normal, _ = delocalize_rotate(to_hit, local_dir)
                    ret_pos   = center + normal * radius
                    ret_pdf   = 0.5 * pdf
                diff      = hit_pos - ret_pos
                dot_light = ti.math.dot(diff.normalized(), normal)
            else:               # sample from pre-defined basis plane
                rand_axis1  = ti.random(float) - 0.5
                rand_axis2  = ti.random(float) - 0.5
                v_axis1     = self.base_1 * self.l1 * rand_axis1
                v_axis2     = self.base_2 * self.l2 * rand_axis2
                ret_pos    += (v_axis1 + v_axis2)
                diff        = hit_pos - ret_pos
                dot_light   = ti.math.dot(diff.normalized(), self.dirv)
            if dot_light <= 0.0:
                ret_int = vec3([0, 0, 0])
                ret_pdf = 1.0
            else:
                diff_norm2 = diff.norm_sqr()
                ret_pdf   *= ti.select(dot_light > 0.0, diff_norm2 / dot_light, 0.0)
                ret_int   *= (1. / ti.max(diff_norm2, 1e-5) * dot_light)
        return ret_pos, ret_int, ret_pdf

    @ti.func
    def eval_le(self, inci_dir: vec3, normal: vec3):
        """ Emission evaluation, incid_dir is not normalized """
        ret_int = vec3([0, 0, 0])
        if self._type == 1:
            dot_light = -ti.math.dot(inci_dir.normalized(), normal)
            if dot_light > 0:
                ret_int = self.intensity        # FIXME: this could be a bug
        return ret_int

    @ti.func
    def solid_angle_pdf(self, incid_dir: vec3, normal: vec3, depth: float):
        """ Area PDF converting to solid angle PDF (for hitting a area light) """
        pdf = 0.0
        if self._type == 1:
            # incid dir is ray incident direction (to the area light) 
            dot_res = ti.abs(ti.math.dot(incid_dir, normal))
            pdf = ti.select(dot_res > 1e-7, self.inv_area * ti.pow(depth, 2) / dot_res, 0.0)
        return pdf

class LightSource:
    """
        Sampling function is implemented in Taichi source. Currently:
        Point / Area / Spot / Directional are to be supported
    """
    def __init__(self, base_elem: xet.Element = None):
        if base_elem is not None:
            self.intensity = rgb_parse(base_elem.find("rgb"))
        else:
            print("Warning: default intializer should only be used in testing.")
            self.intensity = np.ones(3, np.float32)
        self.type: str = base_elem.get("type")
        self.id:   str = base_elem.get("id")
        self.inv_area  = 1.0        # inverse area (for non-point emitters, like rect-area or mesh attached emitters)
        self.attached  = False      # Indicated whether the light source is attached to an object (if True, new sampling strategy should be used)

    def export(self) -> TaichiSource:
        """
            Export to taichi
        """
        raise NotImplementedError("Can not call virtual method to be overridden.")

    def __repr__(self):
        return f"<{self.type.capitalize()} light source. Intensity: {self.intensity}. Area: {1. / self.inv_area:.5f}. Attached = {self.attached}>"
