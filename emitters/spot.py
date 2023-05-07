"""
    Spot light source
    @author: Qianyue He
    @date: 2023-4-15
"""

import sys
sys.path.append("..")

import numpy as np
from taichi.math import vec3
import xml.etree.ElementTree as xet

from parsers.general_parser import vec3d_parse, get
from renderer.constants import ZERO_V3, AXIS_Z, INV_PI, DEG2RAD
from emitters.abtract_source import LightSource, TaichiSource, SPOT_SOURCE

class SpotSource(LightSource):
    """
        Spot Light source is a positionally delta source
    """
    def __init__(self, elem: xet.Element = None):
        super().__init__(elem)
        point_elems = elem.findall("point")
        assert(len(point_elems) >= 2)
        self.dir = AXIS_Z
        self.pos = ZERO_V3
        self.half_cos = np.cos(15.0 * DEG2RAD)
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
            if name == "half-angle":
                self.half_cos = np.cos(max(1e-3, get(float_elem, "value", float)) * DEG2RAD)
        self.inv_area = 1.0

    def __repr__(self):
        return f"<{self.type.capitalize()} light source. Intensity: {self.intensity}. Radius: {self.half_cos}, inv area: {self.inv_area}.>"

    def export(self) -> TaichiSource:
        bool_bits = 0x01 + (self.in_free_space << 4)       # positional delta but not directional delta
        return TaichiSource(_type = SPOT_SOURCE, intensity = vec3(self.intensity), inv_area = self.inv_area,
                    pos = vec3(self.pos), dir = vec3(self.dir), r = self.half_cos, bool_bits = bool_bits, emit_time = self.emit_time)
