"""
    Test passing by reference in Taichi
"""

import taichi as ti
import taichi.math as tm
from taichi.math import vec3

ti.init()

@ti.dataclass
class DummyBSDF:
    pos: vec3
    idx: int
    extra: float

    @ti.func
    def test_warp(self, outer: ti.template(), ver: ti.template(), prev: ti.template()):
        r1, r2, r3 = outer.storage[self.idx] + outer.inner[self.idx], self.pos + outer.storage[self.idx], ver.pos + outer.inner[self.idx]
        outer.storage[self.idx] += 1
        ver.extra = r1.sum()
        print(f"PRINT {prev.idx}")
        return r1, r2, r3

@ti.data_oriented
class InnerClass:
    def __init__(self) -> None:
        self.inner = ti.Vector.field(3, float, 8)

        for i in range(8):
            self.inner[i] = vec3([i, i, i])

@ti.data_oriented
class OuterClass(InnerClass):
    def __init__(self) -> None:
        super().__init__()
        self.storage = ti.Vector.field(3, float, 8)
        for i in range(8):
            self.storage[i] = vec3([i * 10, i * 10, i * 10])

        self.bsdf_field = DummyBSDF.field()
        ti.root.dense(ti.i, 8).place(self.bsdf_field)
        for i in range(8):
            self.bsdf_field[i] = DummyBSDF(pos = vec3([0.1 * float(i), 0.1 * float(i), 0.1 * float(i)]), idx = i)

    @ti.func
    def ref_self_call(self, idx, prev: ti.template()):
        return self.bsdf_field[idx].test_warp(self, self.bsdf_field[(idx + 1) % 8], prev)
    
outer = OuterClass()

null = DummyBSDF(idx = -1)

@ti.kernel
def ref_test():
    for i in outer.bsdf_field:
        v1, v2, v3 = outer.ref_self_call(i, null)
        print(f"Idx: {i}", v1, v2, v3)
        print(f"Idx: {i}, storage = ", outer.storage[i], ", extra: ", outer.bsdf_field[i].extra)
for i in range(8):
    print(f"Outer[{i}].extra = {outer.bsdf_field[i].extra}")

ref_test()

for i in range(8):
    print(f"Outer[{i}].extra = {outer.bsdf_field[i].extra}")
