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
from renderer.constants import INV_PI, ZERO_V3, AXIS_Y
from sampler.general_sampling import sample_triangle, cosine_hemisphere, uniform_sphere

@ti.dataclass
class TaichiSource:
    """
        This implementation is not elegant, too obese. Of course, there are better ways \\
        For example, for every 'variable field', we can place them to bit-masked: \\
        node = pointer -> bitmasked \\
        node.place(pos); node.place(dirv); node.place(base_1); node.place(base_2); \\
        The following implementation is much simpler, and light source will not consume too much memory
    """
    _type:      int      # 0 Point, 1 Area, 2 Spot, 3 Directional
    obj_ref_id: int      # Referring to the attaching object
    bool_bits:  int      # indicate whether the source is a delta source / inside free space
    intensity:  vec3
    pos:        vec3
    inv_area:   float      # inverse area (for non-point emitters, like rect-area or mesh attached emitters)

    @ti.func
    def is_delta_source(self):
        return self.bool_bits & 0x01
    
    @ti.func
    def in_free_space(self):
        return self.bool_bits & 0x02

    @ti.func
    def distance_attenuate(self, x: vec3):
        # This falloff function is... weird
        return ti.min(1.0 / ti.max(x.norm_sqr(), 1e-5), 1.0)
    
    @ti.func
    def sample_hit(
        self, dvs: ti.template(), normals: ti.template(), 
        mesh_cnt: ti.template(), hit_pos: vec3
    ):
        """
            A unified sampling function, choose sampling strategy according to _type \\
            input ray hit point, given the point to illuminate, sample a light source point \\
            returns <sampled source point> <souce intensity> <sample pdf> \\
            sampled PDF is defined on solid angles (but not differential area)
        """
        ret_int = self.intensity
        ret_pos = self.pos
        ret_pdf = 1.0
        normal = ZERO_V3
        if self._type == 0:     # point source
            ret_int *= self.distance_attenuate(hit_pos - ret_pos)
        elif self._type == 1:   # area source
            ret_pdf     = self.inv_area
            dot_light   = 1.0
            diff        = ZERO_V3
            mesh_num = mesh_cnt[self.obj_ref_id]
            normal   = AXIS_Y
            if mesh_num:
                tri_id    = ti.random(int) % mesh_num       # ASSUME that triangles are similar in terms of area
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
                # We only choose the hemisphere, therefore we have a 0.5. Also, this is both sa & area measure
                ret_pdf   = 0.5 * pdf
            diff      = hit_pos - ret_pos
            dot_light = ti.math.dot(diff.normalized(), normal)
            if dot_light <= 0.0:
                ret_int = ZERO_V3
                ret_pdf = 1.0
            else:
                diff_norm2 = diff.norm_sqr()
                ret_pdf *= ti.select(dot_light > 0.0, diff_norm2 / dot_light, 0.0)
                ret_int = ti.select(ret_pdf > 0.0, ret_int / ret_pdf, 0.)
        return ret_pos, ret_int, ret_pdf, normal
    
    @ti.func
    def sample_le(self, dvs: ti.template(), normals: ti.template(), mesh_cnt: ti.template()):
        """ Sample a point on the emitter and sample a random ray without ref point """
        ray_o   = ZERO_V3
        ray_d   = AXIS_Y
        normal  = AXIS_Y
        pdf_dir = 0.
        pdf_pos = 1.
        if self._type == 0:
            # Uniform sampling the sphere, since its uniform, we don't have to set its frame
            ray_d, pdf_dir = uniform_sphere()
            ray_o = self.pos
            normal = ray_d
        elif self._type == 1:       # sampling cosine hemisphere for a given point
            mesh_num = mesh_cnt[self.obj_ref_id]
            if mesh_num:
                tri_id = ti.random(int) % mesh_num       # ASSUME that triangles are similar in terms of area
                normal = normals[self.obj_ref_id, tri_id]
                dv1    = dvs[self.obj_ref_id, tri_id, 0]
                dv2    = dvs[self.obj_ref_id, tri_id, 1]
                ray_o  = sample_triangle(dv1, dv2) + dvs[self.obj_ref_id, tri_id, 2]
            else:
                normal, _p = uniform_sphere()
                center = dvs[self.obj_ref_id, 0, 0]
                radius = dvs[self.obj_ref_id, 0, 1][0]
                ray_o  = center + normal * radius
            local_d, pdf_dir = cosine_hemisphere()
            ray_d, _R = delocalize_rotate(normal, local_d)
            pdf_pos = self.inv_area
        return self.intensity, ray_o, ray_d, pdf_pos, pdf_dir, normal

    @ti.func
    def eval_le(self, inci_dir: vec3, normal: vec3):
        """ Emission evaluation, incid_dir is not normalized """
        ret_int = ZERO_V3
        if self._type == 1:
            dot_light = -ti.math.dot(inci_dir.normalized(), normal)
            if dot_light > 0:
                ret_int = self.intensity    # radiance will remain unchanged
        return ret_int

    @ti.func
    def solid_angle_pdf(self, incid_dir: vec3, normal: vec3, depth: float):
        """ Area PDF converting to solid angle PDF (for hitting a area light) """
        dot_res = ti.abs(ti.math.dot(incid_dir, normal))
        return ti.select(dot_res > 0.0, self.area_pdf() * ti.pow(depth, 2) / dot_res, 0.0)

    @ti.func
    def area_pdf(self):
        """ Area PDF for hitting a area light, this is the non-converted version of solid_angle_pdf """
        pdf = 0.0
        if self._type == 1:
            pdf = self.inv_area
        return pdf
    
    @ti.func
    def direction_pdf(self, exit_dir, light_n):
        """ Compute solid angle PDF for emitting in certain direction """
        pdf = 0.0
        if self._type == 0:         # uniform sphere PDF
            pdf = INV_PI * 0.25
        elif self._type == 1:       # cosine weighted PDF
            pdf = ti.max(ti.math.dot(exit_dir, light_n), 0.0) * INV_PI
        return pdf

class LightSource:
    """
        Sampling function is implemented in Taichi source. Currently:
        Point / Area / Spot / Directional are to be supported
    """
    def __init__(self, base_elem: xet.Element = None):
        self.intensity = np.ones(3, np.float32)
        if base_elem is not None:
            all_rgbs = base_elem.findall("rgb")
            for rgb_elem in all_rgbs:
                name = rgb_elem.get("name")
                if name == "emission":
                    self.intensity = rgb_parse(rgb_elem)
                elif name == "scaler":
                    self.intensity *= rgb_parse(rgb_elem)
        else:
            print("Warning: default intializer should only be used in testing.")
        self.type: str = base_elem.get("type")
        self.id:   str = base_elem.get("id")
        self.inv_area  = 1.0        # inverse area (for non-point emitters, like rect-area or mesh attached emitters)
        self.attached  = False      # Indicated whether the light source is attached to an object (if True, new sampling strategy should be used)
        self.in_free_space = True   # Whether the light source itself is in free space
        bool_elem = base_elem.find("boolean")
        if bool_elem is not None and bool_elem.get("value").lower() == "false":
            # TODO: note that, if world has scattering medium, all the source are in the scattering medium, unless specified
            self.in_free_space = False

    def export(self) -> TaichiSource:
        """
            Export to taichi
        """
        raise NotImplementedError("Can not call virtual method to be overridden.")

    def __repr__(self):
        return f"<{self.type.capitalize()} light source. Intensity: {self.intensity}. Area: {1. / self.inv_area:.5f}. Attached = {self.attached}>"
