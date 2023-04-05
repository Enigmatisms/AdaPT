"""
    wavefront obj file loader
    @author Qianyue He
    @date 2023.1.19
"""
__all__ = ['load_obj_file', 'apply_transform', 'calculate_surface_area']

import numpy as np
import pywavefront as pwf
import xml.etree.ElementTree as xet

from numpy import ndarray as Arr

# supported_rot_type = ("euler", "quaternion", "angle-axis")

def load_obj_file(path: str, precomp_normal = True, verbose = False):
    """
        Meshes (output) are of shape (N_faces, 3, 3)
        Normals should be of shape (N_faces, 3)
    """
    obj         = pwf.Wavefront(path, collect_faces = True)
    vertices    = np.float32(obj.vertices)
    faces       = []
    for mesh in obj.mesh_list:
        for face_idx in mesh.faces:
            faces.append(vertices[face_idx])
    mesh_faces = np.stack(faces, axis = 0)
    normals = None
    if precomp_normal:
        normals = []
        for face_mesh in mesh_faces:
            dp1 = face_mesh[1] - face_mesh[0]
            dp2 = face_mesh[2] - face_mesh[1]
            normal = np.cross(dp1, dp2)         # preserves right-hand order
            normal /= np.linalg.norm(normal)
            normals.append(normal)
        normals = np.stack(normals, axis = 0)
    if verbose:
        print(f"Mesh loaded from {path}, output shape: {mesh_faces.shape}. Normal output: {precomp_normal}")
    return mesh_faces, normals

def calculate_surface_area(meshes: Arr, _type = 0):
    area_sum = 0.
    if _type == 0:
        for face in meshes:
            dv1 = face[1] - face[0]
            dv2 = face[2] - face[0]
            area_sum += np.linalg.norm(np.cross(dv1, dv2)) / 2.
    elif _type == 1:
        radius = meshes[0, 1, 0]
        # ellipsoid surface area approximation
        area_sum = 4. * np.pi * radius ** 2
    return area_sum

def apply_transform(meshes: Arr, normals: Arr, trans_r: Arr, trans_t: Arr) -> Arr:
    """
        - input normals are of shape (N, 3)
        - input meshes are of shape (N, 3, 3), and for the last two dims
            - 3(front): entry index for the vertices of triangles
            - 3(back): entry index for (x, y, z)
    """
    if trans_r is not None:
        center  = meshes.mean(axis = 1).mean(axis = 0)
        meshes -= center                # decentralize
        meshes = meshes @ trans_r      # right multiplication
        if normals is not None: 
            normals = normals @ trans_r # unit normn is preserved
        meshes += center
    if trans_t is not None:
        meshes += trans_t
    return meshes, normals
