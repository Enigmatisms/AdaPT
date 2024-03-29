"""
    Texture definition for python and Taichi
    Current texture implementation is not good: 
    - fixed 1024 * 1024 images
    - checker board texture
    @author: Qianyue He
    @date: 2023-5-10
"""

import os
import json
import cv2 as cv
import numpy as np
import taichi as ti
import taichi.math as tm
import xml.etree.ElementTree as xet

from parsers.general_parser import rgb_parse, get
from taichi.math import vec3
from rich.console import Console
from renderer.constants import ZERO_V3
CONSOLE = Console(width = 128)


"""
    FIXME: to be deleted. This is for simple design consideration
    (1) textures can be loaded from images or use a checkerboard
    (2) detailed settings like scale and checker colors can be done in xml file
    (3) query is based on bilinear interpolation

    FIXME: texture should be part of the BRDF
"""

class Texture_np:
    MODE_IMAGE   = 0
    MODE_CHECKER = 1
    def __init__(self, elem: xet.Element, max_size = 2048) -> None:
        self.tag = elem.get("tag", "albedo")
        self.max_size = max_size
        self.id = elem.get("id")
        self.type = elem.get("type")
        self.c1 = np.zeros(3)
        self.c2 = np.ones(3)
        self.scale_u = 1.
        self.scale_v = 1.
        self.off_x = 0
        self.off_y = 0
        self.h, self.w = 0, 0
        if self.type == "checkerboard":
            self.mode = Texture_np.MODE_CHECKER
            # load checkboard config from xml
            rgb_nodes = elem.findall("rgb")
            num_rgb = len(rgb_nodes)
            if num_rgb > 0:
                self.c1 = rgb_parse(rgb_nodes[0])
                if num_rgb > 1:
                    self.c2 = rgb_parse(rgb_nodes[1])
        else:
            self.mode = Texture_np.MODE_IMAGE
            file_path = elem.find("string").get("value")
            if not os.path.exists(file_path):
                raise ValueError(f"Texture image input path '{file_path}' does not exist.")
            else:
                self.texture_path = file_path
                self.texture_img  = None
                texture_img = cv.imread(file_path)
                texture_img = cv.cvtColor(texture_img, cv.COLOR_BGR2RGB)
                self.h, self.w, _ = texture_img.shape

                if self.h > max_size or self.w > max_size:
                    self.w = min(self.w, max_size)
                    self.h = min(self.h, max_size)
                    texture_img = cv.resize(texture_img, (self.w, self.h))
                self.texture_img = texture_img.astype(np.float32) / 255.
                if self.tag == "bump":
                    # since normally, the up-axis for bump map texture is z, but for AdaPT, the axis should be y
                    self.texture_img[..., [1, 2]] = self.texture_img[..., [2, 1]]
        float_nodes = elem.findall("float")
        for float_n in float_nodes:
            name = float_n.get("name")
            if name in {"scale_u", "scale_v"}:
                self.__setattr__(name, get(float_n, "value"))
            else:
                CONSOLE.log(f"[yellow]:warning: Warning: <{name}> not used in loading textures")

    def export(self):
        _type = Texture_np.MODE_CHECKER if self.type == "checkerboard" else Texture_np.MODE_IMAGE
        return Texture(type = _type, off_x = self.off_x, off_y = self.off_y, w = self.w, h = self.h,
            scale_u = self.scale_u, scale_v = self.scale_v, color1 = self.c1, color2 = self.c2
        )
    
    @staticmethod
    def default():
        return Texture(type = -255, off_x = 0, off_y = 0, w = 0, h = 0,
            scale_u = 1, scale_v = 1, color1 = ZERO_V3, color2 = ZERO_V3
        )

    def __repr__(self) -> str:
        return f"<Texture '{self.id}': {self.off_x}, {self.off_y}, {self.w}, {self.h}>"

# TODO: checkboard UV is not implemented
@ti.dataclass
class Texture:
    type:       int     # id to field represented texture, -1 means checker board -255 means invalid
    off_x:      int     # offset for x axis (in the packed texture image)
    off_y:      int     # offset for y axis (in the packed texture image)
    w:          int     # query will be scaled (bilerped) to (w, h)
    h:          int
    scale_u:    float   # scale factor for u axis: the uv coordinates of a vertex will be multiplied by this
    scale_v:    float   # scale factor for v axis
    color1:     vec3    # for checker board
    color2:     vec3

    @ti.func
    def query(self, textures: ti.template(), u: float, v: float):
        """ u, v is uv_coordinates, which might be fetched by triangle barycentric coords (bi-linear interpolation)
        """
        scaled_u = (u * self.scale_u * self.w).__mod__(self.w - 1.)
        scaled_v = (v * self.scale_v * self.h).__mod__(self.h - 1.)
        floor_u = tm.floor(scaled_u, float)
        floor_v = tm.floor(scaled_v, float)
        ratio_u = scaled_u - floor_u
        ratio_v = scaled_v - floor_v

        floor_u = floor_u + self.off_x
        floor_v = floor_v + self.off_y
        floor_ui = int(floor_u)
        floor_vi = int(floor_v)
        ceil_ui  = floor_ui + 1
        ceil_vi  = floor_vi + 1

        # lerp
        q_ff = textures[floor_vi, floor_ui]
        q_cf = textures[floor_vi, ceil_ui]
        q_fc = textures[ceil_vi, floor_ui]
        q_cc = textures[ceil_vi, ceil_ui]
        mix_1 = tm.mix(q_ff, q_cf, ratio_u)
        mix_2 = tm.mix(q_fc, q_cc, ratio_u)
        return tm.mix(mix_1, mix_2, ratio_v)
