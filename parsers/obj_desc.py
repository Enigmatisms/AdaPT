"""
    Object descriptor (simple)
    @date 2023.1.20
"""

import numpy as np
from numpy import ndarray as Arr

def get_aabb(meshes: Arr, _type: int = 0) -> Arr:
    """
        Axis-aligned bounding box for one object
        input: meshes of shape (N, 3, 3), output: two 3D point describing an AABB
    """
    if _type == 0:
        mini = meshes.min(axis = 1).min(axis = 0)
        maxi = meshes.max(axis = 1).max(axis = 0)
        large_diff = np.abs(maxi - mini) > 1e-3
        for i in range(3):
            if not large_diff[i]:       # special processing for co-plane point AABB
                mini[i] -= 2e-2         # extend 2D AABB (due to being co-plane) to 3D
                maxi[i] += 2e-2
    else:           # extended sphere, which only requires a 6D vector
        mini = meshes[0, 0] - meshes[0, 1]
        maxi = meshes[0, 0] + meshes[0, 1]
    return np.float32((mini, maxi))


class ObjDescriptor:
    def __init__(
        self, meshes, normals, mixture, 
        vert_normal = None,
        uv_coords = None,
        texture_group = None, 
        R = None, t = None, 
        emit_id = -1, _type = 0
    ):
        """
            Inputs are objects on which transformations have been applied already
            - emit_id: if not zero: meaning that this object has an attached emitter
            - _type: if 0: meshes, 1: extended sphere
        """
        self.tri_num = meshes.shape[0]
        self.meshes = meshes
        self.uv_coords = uv_coords
        self.normals = normals
        self.vns = vert_normal
        self.R = R
        self.t = t
        # We do store BSDF in the object descriptor, yet the xml format can remain unchanged
        self.mixture = mixture

        # Texture includes all texture mappings
        self.texture_group = texture_group
        self.aabb = get_aabb(meshes, _type)        # of shape (2, 3)
        self.emitter_ref_id = emit_id
        self.type = _type

    def __repr__(self):
        centroid = (self.aabb[0] + self.aabb[1]) / 2
        if self.type == 0:
            return f"<wavefront with {self.meshes.shape[0]} triangles centered at {centroid}.\n Transformed: {self.R is not None or self.t is not None}>"
        elif self.type == 1:
            return f"<sphere centered at {self.meshes[0, 0]} with radius {self.meshes[0, 1]}>"
        else:
            raise NotImplementedError("Other object types are not supported yet")
