"""
    Collimated light source for transient rendering
    Simulating the behavior of laser light source
    the light source is considered to be an emitter with infinitesimal area
    and the ray from it has only one direction
    @author: Qianyue He
    @date: 2023-3-19 
"""

import sys
sys.path.append("..")

import numpy as np
from taichi.math import vec3
import xml.etree.ElementTree as xet

from scene.general_parser import vec3d_parse, get
from renderer.constants import ZERO_V3, AXIS_Z, INV_PI
from emitters.abtract_source import LightSource, TaichiSource, COLLIMATED_SOURCE


class CollimatedSource(LightSource):
    """
        Collimated light source is a typical delta source.
        But I do implement
        - real-delta: area is infinitesimal --- radius is 0. (cylinder beam cross section radius)
        - pseudo-delta: area is small, but with a valid physical meaning --- radius is positive

        It is impossible to hit (unless the radius is sufficiently large, but this will break the meaning...)
    """
    def __init__(self, elem: xet.Element = None):
        super().__init__(elem)
        point_elems = elem.findall("point")
        assert(len(point_elems) >= 2)
        self.dir = AXIS_Z
        self.pos = ZERO_V3
        self.radius = 0.
        for point_elem in point_elems:
            name = point_elem.get("name")
            if name in {"position", "pos"}:
                self.pos = vec3d_parse(point_elem)
            elif name in {"direction", "dir"}:
                self.dir = vec3d_parse(point_elem)
                norm = np.linalg.norm(self.dir)
                if norm < 1e-5:
                    raise ValueError(f"Direction of collimated source <{self.id}> is ill-conditioned.")
                self.dir /= norm
        float_elems = elem.findall("float")
        for float_elem in float_elems:
            name = float_elem.get("name")
            if name == "radius":
                self.radius = max(0., get(float_elem, "value", float))
        self.inv_area = 1 if self.radius == 0 else INV_PI / (self.radius * self.radius)

    def __repr__(self):
        return f"<{self.type.capitalize()} light source. Intensity: {self.intensity}. Radius: {self.radius}, inv area: {self.inv_area}.>"

    def export(self) -> TaichiSource:
        bool_bits = (self.radius == 0) + 0x02 + (self.in_free_space << 4)       # not necessarily positional-delta but directional delta
        return TaichiSource(_type = COLLIMATED_SOURCE, intensity = vec3(self.intensity), inv_area = self.inv_area,
                    pos = vec3(self.pos), dir = vec3(self.dir), r = self.radius, bool_bits = bool_bits, emit_time = self.emit_time)
