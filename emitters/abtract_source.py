"""
    Light source abstraction
    @date: 2023.1.20
    @author: Qianyue He
"""

import sys
sys.path.append("..")

import numpy as np
import taichi as ti
import taichi.math as tm
import xml.etree.ElementTree as xet
from taichi.math import vec3

from parsers.general_parser import rgb_parse
from renderer.constants import INV_PI, INV_2PI, ZERO_V3, AXIS_Y
from la.cam_transform import delocalize_rotate, world_frame
from sampler.general_sampling import sample_triangle, cosine_hemisphere, uniform_sphere, uniform_cone, concentric_disk_sample

from rich.console import Console
CONSOLE = Console(width = 128)

# point-0: PointSource, area-1: AreaSource, spot-2: SpotSource, collimated-4: CollimatedSource

# =============== Emitter Type =================
POINT_SOURCE        = 0
AREA_SOURCE         = 1
SPOT_SOURCE         = 2
COLLIMATED_SOURCE   = 4

@ti.dataclass
class TaichiSource:
    """
        This implementation is not elegant, too obese. Of course, there are better ways \\
        For example, for every 'variable field', we can place them to bit-masked: \\
        node = pointer -> bitmasked \\
        node.place(pos); node.place(dirv); node.place(base_1); node.place(base_2); \\
        The following implementation is much simpler, and light source will not consume too much memory
    """
    _type:      int         # 0 Point, 1 Area, 2 Spot, 4 collimated
    obj_ref_id: int         # Referring to the attaching object

    # Bool bits: [0 pos delta, 1 dir delta, 2 is area, 3 is inifite, 4 is in free space, 5 is delta, others reserved]
    bool_bits:  int         # indicate whether the source is a delta source / inside free space
    intensity:  vec3
    dir:        vec3        # direction for spot / collimated source
    pos:        vec3
    inv_area:   float       # inverse area (for non-point emitters, like rect-area or mesh attached emitters)
    r:          float       # only useful for collimated source - beam radius, for spot source: half-angle range
    emit_time:  float       # used in transient rendering: when does this emitter start to emit

    @ti.func
    def is_delta_pos(self):             # 0-th bits
        return self.bool_bits & 0x01
    
    @ti.func
    def is_delta_dir(self):             # 1-th bits
        return self.bool_bits & 0x02
    
    @ti.func
    def is_area(self):                  # 2-th bits
        return self.bool_bits & 0x04
    
    @ti.func
    def is_infite(self):                # 3-th bits
        return self.bool_bits & 0x08

    @ti.func
    def in_free_space(self):            # 4-th bits
        return self.bool_bits & 0x10

    @ti.func
    def distance_attenuate(self, x: vec3):
        # This falloff function is... weird
        return ti.min(1.0 / ti.max(x.norm_sqr(), 1e-5), 1.0)
    
    @ti.func
    def sample_hit(
        self, dvs: ti.template(), normals: ti.template(), 
        prim_info: ti.template(), hit_pos: vec3
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
        if self._type == POINT_SOURCE:     # point source
            ret_int *= self.distance_attenuate(hit_pos - ret_pos)
        elif self._type == AREA_SOURCE:   # area source
            ret_pdf     = self.inv_area
            dot_light   = 1.0
            diff        = ZERO_V3
            is_sphere   = prim_info[self.obj_ref_id, 2]
            if is_sphere:
                tri_id    = prim_info[self.obj_ref_id, 0]
                center    = dvs[tri_id, 0]
                radius    = dvs[tri_id, 1][0]
                to_hit    = (hit_pos - center).normalized()
                local_dir, pdf = cosine_hemisphere()
                normal, _ = delocalize_rotate(to_hit, local_dir)
                ret_pos   = center + normal * radius
                # We only choose the hemisphere, therefore we have a 0.5. Also, this is both sa & area measure
                ret_pdf   = 0.5 * pdf
            else:
                mesh_num = prim_info[self.obj_ref_id, 1]
                tri_id    = (ti.random(int) % mesh_num) + prim_info[self.obj_ref_id, 0]  # ASSUME that triangles are similar in terms of area
                normal    = normals[tri_id]
                dv1       = dvs[tri_id, 0]
                dv2       = dvs[tri_id, 1]
                ret_pos   = sample_triangle(dv1, dv2) + dvs[tri_id, 2]
            diff      = hit_pos - ret_pos
            dot_light = tm.dot(diff.normalized(), normal)
            if dot_light <= 0.0:
                ret_int = ZERO_V3
                ret_pdf = 1.0
            else:
                diff_norm2 = diff.norm_sqr()
                ret_pdf *= ti.select(dot_light > 0.0, diff_norm2 / dot_light, 0.0)
                ret_int = ti.select(ret_pdf > 0.0, ret_int / ret_pdf, 0.)
        elif self._type == SPOT_SOURCE:
            to_hit = hit_pos - ret_pos
            depth = ti.max(to_hit.norm(), 1e-5)
            to_hit /= depth                     # normalized
            cos_val = tm.dot(to_hit, self.dir)
            if cos_val > self.r:
                ret_int /= (depth * depth)
            else:
                ret_int.fill(0)
        elif self._type == COLLIMATED_SOURCE:
            # It is possible that we are not able to `sample` (actually, not sample, deterministic calculation) the collimated source
            ret_pdf = 0.
            if self.r > 0.:
                to_hit = (hit_pos - self.pos)
                proj_d = tm.dot(to_hit, self.dir)
                if proj_d > 0.0:
                    dist = ti.sqrt(to_hit.norm_sqr() - proj_d * proj_d)
                    if dist < self.r:
                        # This should not be sampling, since the point on the emitter and the hit point are bijectional
                        ret_pos = hit_pos - proj_d * self.dir       # this would require further intersection test
                        normal = self.dir
                    else:
                        ret_int.fill(0.0)
            else:
                ret_int = ZERO_V3
        return ret_pos, ret_int, ret_pdf, normal
    
    @ti.func
    def sample_le(self, dvs: ti.template(), normals: ti.template(), prim_info: ti.template()):
        """ Sample a point on the emitter and sample a random ray without ref point """
        ray_o   = ZERO_V3
        ray_d   = AXIS_Y
        normal  = AXIS_Y
        pdf_dir = 0.
        pdf_pos = 1.
        if self._type == POINT_SOURCE:
            # Uniform sampling the sphere, since its uniform, we don't have to set its frame
            ray_d, pdf_dir = uniform_sphere()
            ray_o = self.pos
            normal = ray_d
        elif self._type == AREA_SOURCE:       # sampling cosine hemisphere for a given point
            is_sphere = prim_info[self.obj_ref_id, 2]
            if is_sphere:
                tri_id = prim_info[self.obj_ref_id, 0]
                normal, _p = uniform_sphere()
                center = dvs[tri_id, 0]
                radius = dvs[tri_id, 1][0]
                ray_o  = center + normal * radius
            else:
                mesh_num  = prim_info[self.obj_ref_id, 1]
                tri_id    = (ti.random(int) % mesh_num) + prim_info[self.obj_ref_id, 0]       # ASSUME that triangles are similar in terms of area
                normal    = normals[tri_id]
                dv1       = dvs[tri_id, 0]
                dv2       = dvs[tri_id, 1]
                ray_o     = sample_triangle(dv1, dv2) + dvs[tri_id, 2]
            local_d, pdf_dir = cosine_hemisphere()
            ray_d, _R = delocalize_rotate(normal, local_d)
            pdf_pos = self.inv_area
        elif self._type == SPOT_SOURCE:
            local_d = uniform_cone(self.r)
            ray_d, _R = delocalize_rotate(self.dir, local_d)
            pdf_dir = INV_2PI / (1. - self.r)
            ray_o = self.pos
            normal = self.dir
        elif self._type == COLLIMATED_SOURCE:
            # pdf of direction should be delta?
            ray_o = self.pos
            ray_d = self.dir
            normal = ray_d
            pdf_pos = self.inv_area
            pdf_dir = 1.
            if self.r > 0.:
                # sample a point in the disk
                local_offset = concentric_disk_sample() * self.r
                ray_o += world_frame(AXIS_Y, self.dir, local_offset)
        return ray_o, ray_d, pdf_pos, pdf_dir, normal

    @ti.func
    def eval_le(self, inci_dir: vec3, normal: vec3):
        """ Emission evaluation, incid_dir is not normalized """
        ret_int = ZERO_V3
        if self._type == AREA_SOURCE:
            dot_light = -tm.dot(inci_dir.normalized(), normal)
            if dot_light > 0:
                ret_int = self.intensity    # radiance will remain unchanged
        return ret_int

    @ti.func
    def solid_angle_pdf(self, incid_dir: vec3, normal: vec3, depth: float):
        """ Area PDF converting to solid angle PDF (for hitting a area light) """
        dot_res = ti.abs(tm.dot(incid_dir, normal))
        return ti.select(dot_res > 0.0, self.area_pdf() * ti.pow(depth, 2) / dot_res, 0.0)

    @ti.func
    def area_pdf(self):
        """ Area PDF for hitting a area light, this is the non-converted version of solid_angle_pdf """
        pdf = 0.0
        if self._type == AREA_SOURCE:
            pdf = self.inv_area
        return pdf
    
    @ti.func
    def direction_pdf(self, exit_dir, light_n):
        """ Compute solid angle PDF for emitting in certain direction """
        pdf = 0.0
        if self._type == POINT_SOURCE:         # uniform sphere PDF
            pdf = INV_PI * 0.25 
        elif self._type == SPOT_SOURCE:
            pdf = INV_2PI / (1. - self.r)
        elif self._type == AREA_SOURCE:       # cosine weighted PDF
            pdf = ti.max(tm.dot(exit_dir, light_n), 0.0) * INV_PI
        return pdf

class LightSource:
    """
        Sampling function is implemented in Taichi source. Currently:
        Point / Area / Spot / Collimated are to be supported
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
            CONSOLE.log("[yellow]Warning: default intializer should only be used in testing.")
        self.type: str = base_elem.get("type")
        self.id:   str = base_elem.get("id")
        self.inv_area  = 1.0        # inverse area (for non-point emitters, like rect-area or mesh attached emitters)
        self.attached  = False      # Indicated whether the light source is attached to an object (if True, new sampling strategy should be used)
        self.in_free_space = True   # Whether the light source itself is in free space
        self.emit_time = 0.0
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
