"""
    3D space transformation / pinhole camera model
    @author: Qianyue He
    @date: 2023-1-23
"""

import numpy as np
import taichi as ti
import taichi.math as tm
from taichi.math import vec3, mat3
from numpy import ndarray as Arr
from scipy.spatial.transform import Rotation as Rot

__all__ = ['fov2focal', 'np_rotation_between', 'rotation_between', 
           'delocalize_rotate', 'localize_rotate', 'world_frame']

colv3 = ti.types.matrix(3, 1, float)        # column vector
rowv3 = ti.types.matrix(1, 3, float)        # row vector

def fov2focal(fov: float, img_size):
    fov = fov / 180. * np.pi
    return 0.5 * img_size / np.tan(.5 * fov)

@ti.func
def skew_symmetry(vec: vec3):
    return mat3([
        [0, -vec[2], vec[1]], 
        [vec[2], 0, -vec[0]], 
        [-vec[1], vec[0], 0]
    ])
    
def np_rotation_between(fixed: Arr, target: Arr) -> Arr:
    """
        Transform parsed from xml file is merely camera orientation (numpy CPU version)
        INPUT arrays [MUST] be normalized
        Orientation should be transformed to be camera rotation matrix
        Rotation from <fixed> vector to <target> vector, defined by cross product and angle-axis
    """
    axis = np.cross(fixed, target)
    dot = np.dot(fixed, target)
    if abs(dot) > 1. - 1e-5:            # nearly parallel
        return np.sign(dot) * np.eye(3, dtype = np.float32)  
    else:
        # Not in-line, cross product is valid
        axis /= np.linalg.norm(axis)
        axis *= np.arccos(dot)
        euler_vec = Rot.from_rotvec(axis).as_euler('zxy')
        euler_vec[0] = 0                                                # eliminate roll angle
        return Rot.from_euler('zxy', euler_vec).as_matrix()

@ti.func
def rotation_between(fixed: vec3, target: vec3) -> mat3:
    """
        Transform parsed from xml file is merely camera orientation (Taichi version)
        Rodrigues transformation is implemented here
        INPUT arrays [MUST] be normalized
        Orientation should be transformed to be camera rotation matrix
        Rotation from <fixed> vector to <target> vector, defined by cross product and angle-axis
    """
    axis = tm.cross(fixed, target)
    cos_theta = tm.dot(fixed, target)
    ret_R = ti.Matrix.zero(float, 3, 3)
    if ti.abs(cos_theta) < 1. - 1e-5:
        normed_axis = axis.normalized()
        ret_R = ti.Matrix.diag(3, cos_theta) + ((1 - cos_theta) * colv3(*normed_axis)) @ rowv3(*normed_axis) + skew_symmetry(axis)
    else:
        ret_R = ti.Matrix.diag(3, tm.sign(cos_theta))
    return ret_R

@ti.func
def delocalize_rotate(anchor: vec3, local_dir: vec3):
    """ From local frame to global frame """
    R = rotation_between(vec3([0, 1, 0]), anchor)
    return R @ local_dir, R

@ti.func
def localize_rotate(anchor: vec3, global_dir: vec3):
    """ From global frame to local frame """
    R = rotation_between(anchor, vec3([0, 1, 0]))
    return R @ global_dir

@ti.func
def world_frame(local_anchor, global_anchor, local_dir):
    """ Transform from local frame to global frame """
    R = rotation_between(local_anchor, global_anchor)
    return R @ local_dir

if __name__ == "__main__":
    from time import time
    ti.init(kernel_profiler = True)
    print("Numpy / Taichi version rotation computation comparison test")
    print("Numpy CPU version is based on scipy.spatial.transform.Rotation")
    print("Taichi version is based on Rodrigues transformation implemented by me")
    @ti.kernel
    def test_rot(v1_t: vec3, v2_t: vec3) -> mat3:
        return rotation_between(v1_t, v2_t)
    max_cnt = 20000
    valid_cnt = 0
    first_test = True
    total_time = 0.0
    for _ in range(max_cnt):
        v1 = np.random.normal(0, 1, (3,))
        v1 /= np.linalg.norm(v1)
        v2 = np.random.normal(0, 1, (3,))
        v2 /= np.linalg.norm(v2)
        R1 = np_rotation_between(v1, v2)
        if not first_test:
            start_time = time()
        R2 = test_rot(vec3(v1), vec3(v2)).to_numpy()
        if not first_test:
            total_time += time() - start_time
        if first_test:
            first_test = False
        diff = np.abs((R1 - R2))
        if diff.max() < 1e-6:
            valid_cnt += 1
        else:
            print(f"Different in result: {diff.max():9f}, {diff.mean():.9f}")
    print(f"Test samples {max_cnt} in total with {valid_cnt} correct values") 
    print(f"Taichi version takes {total_time:6f} s to run {max_cnt - 1} times")
    print(f"Average computation time: {total_time / (max_cnt - 1) * 1e3:6f} ms")
    ti.profiler.print_kernel_profiler_info() 
    # Average time: 0.185ms, I think this is too slow
