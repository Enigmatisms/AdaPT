"""
    wavefront obj file loader
    @author Qianyue He
    @date 2023.1.19
"""
__all__ = ['extract_obj_info', 'apply_transform', 'calculate_surface_area']

import numpy as np
import pywavefront as pwf

from numpy import ndarray as Arr

from rich.console import Console
CONSOLE = Console(width = 128)

# supported_rot_type = ("euler", "quaternion", "angle-axis")

def extract_obj_info(path: str, verbose = True, auto_scale_uv = True, vn_check = True):
    """ Extract uv coordinates from wavefront object file
        result in UV coordinates not exactly within [0, 1], therefore we might want
        to scale it to make the subsequent processing easier.
        
        Since in my Taichi implementation, primitives are stored in ti.vec3.field (shape (N, 3))
        I reorder the UV vertices into (N, 3)

        auto_scale_uv is True: by default, since the spherical projection might
        vn_check: we can load precomputed normals from obj file, but they are actually vertex normals
        What we need are surface normals, so we need to check them before exporting 
    """
    material = None
    obj      = pwf.Wavefront(path, collect_faces = True)
    for wrapper in obj.materials.values():
        material = wrapper
        break
    if material is not None:
        vert_type = material.vertex_format
        if "T" not in vert_type:
            if verbose:
                CONSOLE.log(f"[blue]Attention: Object contains no uv-coordinates for vtype '{vert_type}'")
        all_parts = vert_type[:-1].split("F_")
        start_dim = 0
        dim_num = sum([int(part[1:]) for part in all_parts])
        all_data = np.float32(material.vertices).reshape(-1, dim_num)
        mesh_faces = None
        uv_coords = None
        normals = None
        has_vn = False
        for part in all_parts:
            if part.startswith("T"):
                uv_coords = all_data[:, start_dim:start_dim+2]
                if auto_scale_uv:
                    result_min = uv_coords.min()
                    result_max = uv_coords.max()
                    uv_coords = (uv_coords - result_min) / (result_max - result_min)
                uv_coords = uv_coords.reshape(-1, 3, 2)
            elif part.startswith("N"):
                has_vn = True
                normals = np.float32(all_data[:, start_dim:start_dim+3]).reshape(-1, 3, 3)
            elif part.startswith("V"):
                mesh_faces = np.float32(all_data[:, start_dim:start_dim+3]).reshape(-1, 3, 3)
            start_dim += int(part[1:])

        assert mesh_faces is not None     # we directly use the vertices loaded, so this can not be empty
        # obj.vertices is not organized, material.vertices is organized
        # therefore, using material.vertices will boost loading speed
        # so uv-coordinates are ordered too
        # vertices shape: (N_faces, 3, 3), uv_coords shape: (N_faces, 3, 2)

        if has_vn and vn_check:         # check 20% vns
            num_vn = normals.shape[0]
            for i in range(0, num_vn, 5):
                if abs(np.linalg.det(normals[i])) > 1e-4:       # determinant check
                    if verbose:
                        CONSOLE.log("Vertex normals are used, loaded from .obj file.")
                    break
            else:
                normals = None
        if normals is None:                 # normal is not computed in obj
            dp1 = mesh_faces[:, 1, :] - mesh_faces[:, 0, :]
            dp2 = mesh_faces[:, 2, :] - mesh_faces[:, 1, :]
            normals = np.cross(dp1, dp2)
            normals /= np.linalg.norm(normals, axis = -1, keepdims = True)
        
        if verbose:
            CONSOLE.log(f"Mesh loaded from '{path}', output shape: [blue]{mesh_faces.shape}[/blue]")
        return mesh_faces, normals, uv_coords
    else:
        raise ValueError("This wavefront onject file has no material but it is required.")

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
