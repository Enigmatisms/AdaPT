"""
    Test some of the available acceleration schemes
"""

import numpy as np
import taichi as ti
from taichi.math import vec3

ti.init(kernel_profiler = True, arch = ti.cuda)

@ti.dataclass
class Vertex:
    pos: vec3
    vec: vec3
    fp: float
    type: int

vertices = Vertex.field()
node_handle = ti.root.dense(ti.i, 16384)
node_handle.dense(ti.j, 64).place(vertices)

vec_field = vec3.field()
node_handle.dense(ti.j, 64).place(vec_field)

@ti.kernel
def initialize():
    for i, j in vertices:
        vertices[i, j] = Vertex(
            fp = (float(i) + float(j)).__mod__(2.6),
            pos = vec3([i, j, 0]), vec = vec3([-i, -j, 0]), type = (i + j) % 2)


@ti.kernel
def compute():
    ti.loop_config(parallelize=8, block_dim=64)
    ti.cache_read_only(vertices)
    for i, j in vertices:
        vertex = vertices[i, j]
        if vertex.type == 0:
            vec_field[i, j] = vertex.pos + vertex.vec
            norm1 = vertex.pos.norm()
            norm2 =  vertex.vec.norm()
            vec_field[i, j] += vec3([0, 0, norm1 + norm2])
            vertices[i, j].pos += 1
        else:
            vec_field[i, j] = vertex.pos - vertex.vec
            norm1 = vertex.pos.norm()
            norm2 =  vertex.vec.norm()
            vec_field[i, j] += vec3([0, 0, norm1 + norm2 * 0.5])

initialize()
for i in range(200):
    compute()

ti.profiler.print_kernel_profiler_info() 