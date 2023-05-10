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
import taichi.types as ttype
from taichi.math import vec3
from rich.console import Console
CONSOLE = Console(width = 128)

MODE_IMAGE = 0
MODE_CHECKER = 1

"""
    FIXME: to be deleted. This is for simple design consideration
    (1) textures can be loaded from images or use a checkerboard
    (2) detailed settings like scale and checker colors can be done in xml file
    (3) query is based on bilinear interpolation
"""

class Texture_np:
    def __init__(self, path: str, max_size = 1024) -> None:
        if not os.path.exists(path):
            CONSOLE.log(f"[yellow]:warning: Warning: texture from '{path}' does not exist. Returing None")
        if path is None:
            self.mode = MODE_CHECKER
            with open(path, 'r', encoding = 'utf-8') as file:
                self.json_data = json.load(file)
        else:
            self.mode = MODE_IMAGE
            self.texture_path = path
            self.texture_img  = None
            texture_img = cv.imread(path)
            texture_img = cv.cvtColor(texture_img, cv.COLOR_BGR2RGB)
            self.h, self.w, _ = texture_img.shape

            if self.h > max_size or self.w > max_size:
                self.w = min(self.w, max_size)
                self.h = min(self.h, max_size)
                texture_img = cv.resize(texture_img, (self.w, self.h))
            self.texture_img = texture_img

    def export(self):
        pass

@ti.dataclass
class Texture:
    id: int         # id to field represented texture, -1 means checker board
    w: float        # query will be scaled (bilerped) to (w, h)
    h: float
    scale_u: float  # scale factor for u axis: the uv coordinates of a vertex will be multiplied by this
    scale_v: float  # scale factor for v axis
    color1: vec3    # for checker board
    color2: vec3

    @ti.func
    def query(self, textures: ti.template, obj_id: int, u: float, v: float):
        """ u, v is uv_coordinates, which might be fetched by triangle barycentric coords (lerp)
            TODO: maybe I should use (pointer) + (dynamic) field here to represent texture with different shape 
        """
        img = textures[obj_id]
        scaled_u = (u * self.scale_u).__mod__(self.w - 1.)
        scaled_v = (v * self.scale_v).__mod__(self.h - 1.)
        floor_u = tm.floor(scaled_u, float)
        floor_v = tm.floor(scaled_v, float)
        ceil_u  = floor_u + 1.
        ceil_v  = floor_v + 1.

        pass


