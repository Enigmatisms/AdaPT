"""
    Various kinds of low level element parsers
    @date: 2023.1.20
    @author: Qianyue He
"""

import numpy as np
from typing import Tuple
from numpy import ndarray as Arr
import xml.etree.ElementTree as xet
from scipy.spatial.transform import Rotation as Rot

def get(node: xet.Element, name: str, _type = float):
    # "0" serves as the default value to be converted (0, 0.0, False)
    return _type(node.get(name, "0"))

def parse_str(val_str: str, no_else_branch = False) -> Arr:
    splitter = (',', ' ')
    for split in splitter:
        if split in val_str:
            all_parts = val_str.split(split)
            return np.float32([float(part.strip()) for part in all_parts])
    else:       # single scalar marked as RGB
        if no_else_branch:
            raise ValueError("Value can not be a single digit, should be a vector splitted by ',' or [space]")
        return np.float32([float(val_str.strip())] * 3)

def rgb_parse(elem: xet.Element):
    if elem is None:
        raise ValueError("EmptyElementError: Element <RGB> is None.")
    else:
        val_str = elem.get("value")
        if val_str is None:
            if elem.get("r"):
                return np.float32([get(elem, "r"), get(elem, "g"), get(elem, "b")])
            else:
                raise ValueError("RGBError: RGB element does not contain valid field.")
        else:
            if val_str.startswith("#"): # html-like hexidecimal RGB
                rgb = np.zeros(3, dtype = np.float32)
                for i in range(3):
                    base = 1 + (i << 1)
                    rgb[i] = int(val_str[base:base + 2], 16) / 255.
                return rgb
            else:
                return parse_str(val_str)

def vec3d_parse(elem: xet.Element):
    if elem.tag == "point":
        if elem.find("value") is None:
            return np.float32([get(elem, "x"), get(elem, "y"), get(elem, "z")])
        else:
            # for positions, implicit float -> vec3 conversion is not allowed
            return parse_str(elem.get("value"), no_else_branch = True)  

def transform_parse(transform_elem: xet.Element) -> Tuple[Arr, Arr, Arr]:
    """
        Note that: extrinsic rotation is not supported, 
        meaning that we can only rotate around the object centroid,
        which is [intrinsic rotation]. Yet, extrinsic rotation can be composed
        by intrinsic rotation and translation
    """
    trans_r, trans_t, trans_s = None, None, None
    for child in transform_elem:
        if child.tag == "translate":
            trans_t = np.float32([get(child, "x"), get(child, "y"), get(child, "z")])
        elif child.tag == "rotate":
            rot_type = child.get("type", "euler")
            if rot_type == "euler":
                r_angle = get(child, "r")   # roll
                p_angle = get(child, "p")   # pitch
                y_angle = get(child, "y")   # yaw
                trans_r = Rot.from_euler("zxy", (r_angle, p_angle, y_angle), degrees = True).as_matrix()
            elif rot_type == "quaternion":
                trans_r = Rot.from_quat([get(child, "x"), get(child, "y"), get(child, "z"), get(child, "w")]).as_matrix()
            elif rot_type == "angle-axis":
                axis: Arr = np.float32([get(child, "x"), get(child, "y"), get(child, "z")])
                axis /= np.linalg.norm(axis) * get(child, "angle") / 180. * np.pi
                trans_r = Rot.from_rotvec(axis).as_matrix()
            else:
                raise ValueError(f"Unsupported rotation representation '{rot_type}'")
        elif child.tag == "scale":
            trans_s = np.float32([get(child, "x"), get(child, "y"), get(child, "z")])
        elif child.tag.lower() == "lookat":
            target_point = parse_str(child.get("target"))
            origin_point = parse_str(child.get("origin"))
            direction = target_point - origin_point
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-5:
                raise ValueError("Normal length too small: Target and origin seems to be the same point")
            # up in XML field is not usefull (polarization), trans_r being a vector means directional vector
            trans_r = direction / dir_norm
            trans_t = origin_point
        else:
            raise ValueError(f"Unsupported transformation representation '{child.tag}'")
    # Note that, trans_r (rotation) is defualt to be intrinsic (apply under the centroid coordinate)
    # Therefore, do not use trans_r unless you know how to correctly transform objects with it
    return trans_r, trans_t, trans_s         # trans_t, trans_r and trans_s could be None, if <transform> is not defined in the object

def parse_sphere_element(elem: xet.Element):
    sphere_info = np.zeros((1, 2, 3), np.float32)
    sphere_info[0, 0] = vec3d_parse(elem.find("point"))
    radius = get(elem.find("float"), "value")
    sphere_info[0, 1] = np.full((3, ), radius)
    return sphere_info, np.float32([[0, 1, 0]])    # sphere has no explicit normal
