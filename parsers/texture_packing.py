"""
    Packing up the texture images into a larger texture image
    for Taichi to load into the field (since texture images with various size
    can be difficult to tackle with using field)

    TODO: in order to understand packing algorithm (for interview purposes)
    Rect packing algorithm should be implemented using C++ by hand
    Ref: https://www.codeproject.com/Articles/210979/Fast-optimizing-rectangle-packing-algorithm-for-bu

    @author: Qianyue He
    @date: 2023-5-17
"""

import sys
import time
import random
import numpy as np
import rectpack as rtp
import matplotlib.pyplot as plt
from utils.tools import timing

from typing import List, Tuple
from bxdf.texture import Texture_np

from matplotlib.patches import Rectangle

SIZE2USE = [3072, 2048, 1024, 720]

__all__ = ("image_packer")

@timing()
def image_packer(textures: List[Texture_np]) -> Tuple[np.ndarray, List[Texture_np]]:
    # TODO: Check this logic? this seems strange
    starting_point = 3
    total_size = 0
    rects = []
    for idx, texture in enumerate(textures):
        if texture.type == Texture_np.MODE_CHECKER: continue
        h, w, _ = texture.texture_img.shape
        max_size = max(h, w)
        if max_size > 1024:
            starting_point = 0
        elif max_size > 720:
            starting_point = 1
        elif max_size > 400:
            starting_point = 2
        total_size += h * w
        rects.append((w, h, idx))               # but rect_id does
    total_size = np.sqrt(total_size) * 1.1      # 1.1 is for redundancy
    final_size = 1024
    for idx in range(starting_point, -1, -1):
        cur_size = SIZE2USE[idx]
        if total_size > cur_size: continue

        packer, pack_success = get_packed_rects(rects, cur_size)
        if pack_success:
            final_size = cur_size
            for packed_r in packer[0]:
                rid = packed_r.rid
                assert textures[rid].w == packed_r.width and textures[rid].h == packed_r.height 
                textures[rid].off_x = packed_r.x
                textures[rid].off_y = packed_r.y
            break
    else:
        raise ValueError("Texture image packing failed, max size 4096 can not even satisfy.")
    result_image = np.zeros((final_size, final_size, 3), dtype = np.float32)
    result_dict = {}
    for texture in textures:
        sx, sy = texture.off_x, texture.off_y
        ex, ey = sx + texture.w, sy + texture.h
        result_image[sy:ey, sx:ex, :] = texture.texture_img
        result_dict[texture.id] = texture
    return result_image, result_dict

def get_packed_rects(rects: List[Tuple[int, int]], max_size: int = 1024):
    packer = rtp.newPacker(rotation = False)
    packer.add_bin(max_size, max_size)
    for rect in rects:
        packer.add_rect(*rect)
    packer.pack()
    success = len(packer[0]) == len(rects)
    return packer, success

def random_rect_gen(num, min_x = 8, max_x = 9, min_y = 8, max_y = 9, standard = True):
    """ Generate random rectangle for testing
        standard: True by default, if True, w/h of rectangle will be the power of 2
    """
    results = []
    for _ in range(num):
        if standard:
            x = 2 ** random.randint(min_x, max_x)
            y = 2 ** random.randint(min_y, max_y)
        else:
            x = random.randint(min_x, max_x)
            y = random.randint(min_y, max_y)
        results.append((x, y))
    return results

def plot_packed(packer: rtp.PackerBBF, rx = 2048, ry = 2048):
    fig, ax = plt.subplots()
    ax.set_xlim([0, rx])
    ax.set_ylim([0, ry])
    packed = packer[0]
    for rect in packed:
        ax.add_patch(Rectangle((rect.x, rect.y), rect.width, rect.height,
             edgecolor = 'black',
             facecolor = 'blue',
             fill=True,
             lw=1))
    plt.show()

if __name__ == "__main__":
    random.seed(int(sys.argv[1]))
    packer = rtp.newPacker(rotation = False)
    packer.add_bin(2048, 2048)
    
    rects = random_rect_gen(20)
    for i, rect in enumerate(rects):
        packer.add_rect(*rect, rid = i)

    start_time = time.time()
    packer.pack()
    end_time = time.time()
    print(f"Packing time consumption: {end_time - start_time} s\n")

    print("Original rect:")
    for rect in rects:
        print(f"R({rect[0]}, {rect[1]})")
    print("")

    for rect in packer[0]:
        print(rect, rect.rid)
    plot_packed(packer)

