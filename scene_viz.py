"""
    Scene visualizer for FoV and pose setting
    @author: Qianyue He
    @date: 2023-2-18
"""
import os
import sys
sys.path.append("..")

import glfw
import numpy as np
import taichi as ti
import taichi.ui as tui
import taichi.math as tm
from taichi.math import vec3, mat3

from typing import List
from la.cam_transform import *
from scipy.spatial.transform import Rotation as Rot
from tracer.tracer_base import TracerBase

from parsers.obj_desc import ObjDescriptor
from parsers.xml_parser import scene_parsing
from parsers.opts import get_options
from utils.tools import folder_path

"""
    Todo: implement better viewing methods
"""

MAX_HEIGHT = 1024
MAX_WIDTH  = 1024

@ti.data_oriented
class Visualizer(TracerBase):
    """ Emitter is not supported. We only import mesh / sphere object in here """
    def __init__(self, objects: List[ObjDescriptor], prop: dict):
        self.w          = ti.field(ti.i32, ())
        self.h          = ti.field(ti.i32, ())
        self.focal      = ti.field(ti.f32, ())
        self.inv_focal  = ti.field(ti.f32, ())
        self.w[None]    = prop['film']['width']                              # image is a standard square
        self.h[None]    = prop['film']['height']

        self.focal[None]     = fov2focal(prop['fov'], min(self.w[None], self.h[None]))
        self.inv_focal[None] = 1. / self.focal[None]

        self.num_objects = len(objects)
        self.num_prims = sum([obj.tri_num for obj in objects])

        self.cam_orient = ti.Vector.field(3, float, ())
        self.cam_t      = ti.Vector.field(3, float, ())
        self.cam_r      = ti.Matrix.field(3, 3, float, ())

        cam_orient            = prop['transform'][0]
        cam_orient            /= np.linalg.norm(cam_orient)
        self.cam_orient[None] = vec3(cam_orient)
        self.cam_t[None]      = vec3(prop['transform'][1])
        self.cam_r[None]      = mat3(np_rotation_between(np.float32([0, 0, 1]), cam_orient))
        
        self.aabbs      = ti.Vector.field(3, float, (self.num_objects, 2))
        self.normals    = ti.Vector.field(3, float)
        self.prims      = ti.Vector.field(3, float)                             # leveraging SSDS, shape (N, mesh_num, 3) - vector3d
        self.precom_vec = ti.Vector.field(3, float)
        self.pixels     = ti.Vector.field(3, float, (1024, 1024))      # maximum size: 1024

        self.dense_nodes = ti.root.dense(ti.i, self.num_prims)
        self.dense_nodes.place(self.normals)
        self.dense_nodes.dense(ti.j, 3).place(self.prims, self.precom_vec)      # for simple shapes, this would be efficient

        # pos0: start_idx, pos1: number of primitives, pos2: obj_id (being triangle / sphere? Others to be added, like cylinder, etc.)
        self.obj_info  = ti.field(int, (self.num_objects, 3))
        self.load_primitives(**array_info)
        self.initialze(objects)

    def load_primitives(
        self, primitives: np.ndarray, indices: np.ndarray, 
        n_g: np.ndarray, n_s: np.ndarray, uvs: np.ndarray
    ):
        """ Load primitives via faster API """
        self.prims.from_numpy(primitives)
        # sphere primitives are padded
        prim_vecs = np.concatenate([
            primitives[..., 1, :] - primitives[..., 0, :],
            primitives[..., 2, :] - primitives[..., 0, :],
            primitives[..., 0, :]], axis = -2)
        prim_vecs[indices, :2, :] = primitives[indices, :2, :]
        self.precom_vec.from_numpy(prim_vecs)
        self.normals.from_numpy(n_g)

    def set_width(self, val: int):
        self.w[None] = int(val)
    
    def set_height(self, val: int):
        self.h[None] = int(val)

    def set_translation(self, t: np.ndarray):
        self.cam_t[None] = vec3(t)

    def set_rotation(self, rpy: np.ndarray):
        trans_r = Rot.from_euler("zxy", rpy, degrees = True).as_matrix()
        self.cam_r[None] = mat3(trans_r)

    def local_to_global(self):
        forward = self.cam_r[None] @ vec3([0, 0, 1])        # local z direction to global
        lateral = self.cam_r[None] @ vec3([1, 0, 0])        # local x direction to global
        elevate = self.cam_r[None] @ vec3([0, 1, 0])        # local y direction to global
        return forward, lateral, elevate

    def initialze(self, objects: List[ObjDescriptor]):
        acc_prim_num = 0
        for i, obj in enumerate(objects):
            self.obj_info[i, 0] = acc_prim_num
            self.obj_info[i, 1] = obj.tri_num
            self.obj_info[i, 2] = obj.type
            acc_prim_num        += obj.tri_num

            self.aabbs[i, 0]    = vec3(obj.aabb[0])        # unrolled
            self.aabbs[i, 1]    = vec3(obj.aabb[1])

    @ti.func
    def pix2ray(self, i, j):
        inv_focal = self.inv_focal[None]
        cam_dir = vec3([(512.5 - float(i)) * inv_focal, (float(j) - 512.5) * inv_focal, 1.])
        return (self.cam_r[None] @ cam_dir).normalized()

    def calculate_focal(self, fov):
        self.focal[None]     = fov2focal(fov, min(self.w[None], self.h[None]))
        self.inv_focal[None] = 1. / self.focal[None]

    @ti.kernel
    def render(self):
        for i, j in self.pixels:
            ray = self.pix2ray(i, j)
            obj_id, normal, _d, _u, _v = self.ray_intersect(ray, self.cam_t[None])
            if obj_id >= 0:
                self.pixels[i, j].fill(ti.abs(tm.dot(ray, normal)))
            else:
                self.pixels[i, j].fill(0.0)

def get_points(off_x, off_y):
    start_p = np.float32([[0.5 - off_x, 0.5 - off_y, 0], [0.5 + off_x, 0.5 - off_y, 0],
                [0.5 + off_x, 0.5 + off_y, 0], [0.5 - off_x, 0.5 + off_y, 0]])
    result = np.zeros((8, 3), np.float32)
    for i in range(4):
        result[i << 1] = start_p[i]
        result[(i << 1) + 1] = start_p[(i + 1) % 4]
    return result

def get_translation(gui, tx, ty, tz):
    t_x = gui.slider_float('X', tx, -20., 20.)
    t_y = gui.slider_float('Y', ty, -20., 20.)
    t_z = gui.slider_float('Z', tz, -20., 20.)
    return np.float32([t_x, t_y, t_z])

def get_rotation(gui, r, p, y):
    r = gui.slider_float('Roll', r, -180., 180.)
    p = gui.slider_float('Pitch', p, -180., 180.)
    y = gui.slider_float('Yaw', y, -180., 180.)
    return np.float32([r, p, y])

if __name__ == "__main__":
    glfw.init()
    options = get_options()
    cache_path = folder_path(f"./cached/viz/{options.scene}", f"Cache path for scene {options.scene} not found. JIT compiling...")
    ti.init(arch = ti.vulkan, default_ip = ti.i32, default_fp = ti.f32, offline_cache_file_path = cache_path)
    vertex_field = ti.Vector.field(3, float, 8)
    input_folder = os.path.join(options.input_path, options.scene)
    _, array_info, all_objs, configs = scene_parsing(input_folder, options.name)  # complex_cornell

    viz = Visualizer(all_objs, configs)
    init_R = Rot.from_matrix(viz.cam_r[None].to_numpy()).as_euler('zxy', degrees = True)

    # GGUI initializations 
    window   = tui.Window('Scene Interactive Visualizer', res = (1024, 1024), pos = (150, 150))
    canvas   = window.get_canvas()
    gui      = window.get_gui()
    width    = gui.slider_int('Width', configs['film']['width'], 32, 1024)
    height   = gui.slider_int('Height', configs['film']['height'], 32, 1024)
    fov      = gui.slider_float('FoV', configs['fov'], 20., 80.)
    trans_r  = get_rotation(gui, *init_R)
    reset_bt = gui.button('Reset')
    show_pose = gui.button('Show pose')

    last_fov   = fov
    last_w     = width
    last_h     = height
    last_r   = trans_r.copy()
    while window.running:
        if gui.button('Reset'):
            trans_r  = get_rotation(gui, *init_R)
            print(trans_r)
            width    = configs['film']['width']
            height   = configs['film']['height']
            fov      = configs['fov']
        
        if gui.button('Show pose'):
            forward_axis = viz.cam_r[None] @ vec3([0, 0, 1])
            print("Position: viz.cam_t = ", viz.cam_t[None])
            print("Lookat: ", viz.cam_t[None] + forward_axis)
           
        width   = gui.slider_int('Width', width, 32, 1024)
        height  = gui.slider_int('Height', height, 32, 1024)
        fov     = gui.slider_float('FoV', fov, 20., 80.)
        trans_r = get_rotation(gui, *trans_r)
        if abs(fov - last_fov) >= 0.1:
            viz.calculate_focal(fov)
            last_fov = fov
        if width != last_w:
            viz.set_width(width)
            viz.calculate_focal(fov)
            last_w    = width
        if height != last_h:
            viz.set_height(height)
            viz.calculate_focal(fov)
            last_h    = height
        if (last_r - trans_r).any():
            viz.set_rotation(trans_r)
            last_r = trans_r.copy()
        forward, lateral, elevate = viz.local_to_global()
        if   window.is_pressed("w"):        viz.cam_t[None] += 0.2 * forward
        elif window.is_pressed("s"):        viz.cam_t[None] -= 0.2 * forward
        elif window.is_pressed("a"):        viz.cam_t[None] += 0.2 * lateral
        elif window.is_pressed("d"):        viz.cam_t[None] -= 0.2 * lateral
        elif window.is_pressed(tui.SPACE):  viz.cam_t[None] += 0.2 * elevate
        elif window.is_pressed(tui.SHIFT):  viz.cam_t[None] -= 0.2 * elevate
        elif window.is_pressed(tui.ESCAPE): window.running = False
        
        viz.render()
        canvas.set_image(viz.pixels)
        points = get_points(0.5 * width / MAX_WIDTH, 0.5 * height / MAX_HEIGHT)
        for i, pt in enumerate(points):
            vertex_field[i] = vec3(pt)
        canvas.lines(vertex_field, width = 0.002, color = (0., 0.4, 1.0))
        window.show()
        if window.running == False: break
        