import taichi as ti
import taichi.types as ttype

ti.init()

big_field = ti.field(ti.i32, (32, 32))

vec2i = ttype.vector(2, int)
field = ti.Vector.field(2, ti.i32)
node = ti.root.dense(ti.ij, (8, 8))
node.place(field, offset = (6, 7))

@ti.kernel
def setter():
    for i, j in big_field:
        if i >= 6 and i < 14 and j >= 7 and j < 15:
            field[i, j] = vec2i([i, j])

@ti.kernel
def getter():
    for i, j in field:
        print(f"{i}, {j} = ", field[i, j])

setter()
getter()