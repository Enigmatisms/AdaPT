import numpy as np
import taichi as ti
from taichi.math import vec3
import taichi.ui as tui

PI_DIV2 = np.pi / 2.
PI_DIV4 = np.pi / 4.
ZERO_V3 = vec3([0, 0, 0])

@ti.func
def concentric_sample():
    off_x = ti.random(float) * 2. - 1.
    off_y = ti.random(float) * 2. - 1.
    result = ZERO_V3
    if off_x != 0 and off_y != 0:
        if ti.abs(off_x) > ti.abs(off_y):
            theta = PI_DIV4 * (off_y / off_x)
            result = vec3([off_x * ti.cos(theta), 0., off_x * ti.sin(theta)])
        else:
            theta = PI_DIV2 - PI_DIV4 * (off_x / off_y)
            result = vec3([off_y * ti.cos(theta), 0., off_y * ti.sin(theta)])
    return result

@ti.kernel
def get_samples(field: ti.template()):
    for i in field:
        sample = concentric_sample()
        field[i] = vec3([sample[0],sample[2], -1]) * 0.5 + 0.5
        print(field[i])
    print("Samples are obtained.")

if __name__ == '__main__':
    ti.init()
    vec_field = ti.Vector.field(3, float, 10000)
    get_samples(vec_field)
    window   = tui.Window('Scene Interactive Visualizer', res = (1024, 1024), pos = (150, 150))
    canvas   = window.get_canvas()
    while window.running:
        canvas.circles(vec_field, 0.001, color = (0., 0.4, 1.0))
        for e in window.get_events(tui.PRESS):
            if e.key == tui.ESCAPE:
                window.running = False
        window.show()
