"""
    Rectangluar area light source
    @author: Qianyue He
    @date: before 2023.2.1
"""

import sys
sys.path.append("..")

import numpy as np
from taichi.math import vec3
import xml.etree.ElementTree as xet

from emitters.abtract_source import LightSource, TaichiSource
from scene.general_parser import vec3d_parse

class AreaSource(LightSource):
    def __init__(self, elem: xet.Element):
        """
            This is more complex than other light sources
            The problem is how to export different kinds of light source to Taichi
            Note that: defining area source using two base vectors is deprecated since the subsequent commit of commit 08aa645d 
        """
        super().__init__(elem)
        self.attached = True

    def export(self) -> TaichiSource:
        bool_bits = (self.in_free_space << 4) | 0x04
        return TaichiSource(_type = 1, bool_bits = bool_bits, intensity = vec3(self.intensity), inv_area = self.inv_area, emit_time = self.emit_time)
        