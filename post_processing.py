""" Firefly removal (simple) using conservative median filter
    @author Qianyue He
    @date 2023.8.7
"""

import sys
import numpy as np
import taichi as ti
import matplotlib.pyplot as plt

from taichi.math import vec3
THRESHOLD = 0.4

@ti.kernel
def filtering(src: ti.template(), dst: ti.template()):
    for i, j in dst:
        center_pix = src[i + 1, j + 1]
        valid = False
        pix_val_sum = vec3([0, 0, 0])
        for k_x in range(3):
            for k_y in range(3):
                if k_x == 1 and k_y == 1: continue
                pix = src[i + k_x, j + k_y]
                norm = (pix - center_pix).norm()
                if norm < THRESHOLD:
                    valid = True
                    break
                pix_val_sum += pix
        if valid:
            dst[i, j] = center_pix
        else:
            dst[i, j] = pix_val_sum / 8.

if __name__ == "__main__":
    ti.init(arch = ti.cpu)
    img = plt.imread(sys.argv[1])
    pad_img = np.pad(img, ((1, 1), (1, 1), (0, 0)))
    h, w, _ = img.shape

    img_field = ti.Vector.field(3, float, (h + 2, w + 2))
    out_field = ti.Vector.field(3, float, (h, w))
    img_field.from_numpy(pad_img)
    filtering(img_field, out_field)
    print(f"Image loaded from '{sys.argv[1]}' and exported to './processed.png'")
    plt.imsave("processed.png", out_field.to_numpy())
