"""
    The simplest light source: point source
    @date: 2023.1.20
    @author: Qianyue He
    sampling is not yet implemented (2023.1.20 version)
"""

import sys
sys.path.append("..")

import numpy as np
from taichi.math import vec3
import xml.etree.ElementTree as xet

from emitters.abtract_source import LightSource, TaichiSource
from scene.general_parser import vec3d_parse


class PointSource(LightSource):
    def __init__(self, elem: xet.Element = None):
        super().__init__(elem)
        pos_elem = elem.find("point")
        assert(pos_elem is not None)
        self.pos: np.ndarray = vec3d_parse(pos_elem)

    def export(self) -> TaichiSource:
        bool_bits = 1 + (self.in_free_space << 1)
        return TaichiSource(_type = 0, intensity = vec3(self.intensity), pos = vec3(self.pos), bool_bits = bool_bits)
