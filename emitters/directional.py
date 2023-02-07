"""
    TODO: Directional light source (global paralleled ray light source)
"""

import sys
sys.path.append("..")

import numpy as np
import xml.etree.ElementTree as xet

from emitters.abtract_source import LightSource
from scene.general_parser import vec3d_parse

class DirectionalSource(LightSource):
    def __init__(self, elem: xet.Element):
        super().__init__(elem)
        dir_elem = elem.find("point")
        assert(dir_elem)
        self.direction = vec3d_parse(dir_elem)