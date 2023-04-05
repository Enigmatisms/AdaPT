import taichi as ti
import taichi.math as tm
from taichi.math import vec3

ti.init()

@ti.data_oriented
class DummyBSDF:
    def __init__(self):
        self.a = vec3([0, 0, 0])

    @ti.kernel
    def print(self):
        for i in range(8):
            print(self.a[0], self.a[1], self.a[2])
            self.a[0] += 1
            self.a[1] += 1
            self.a[2] += 1

if __name__ == '__main__':
    ti.init()
    bsdf = DummyBSDF()
    bsdf.print()