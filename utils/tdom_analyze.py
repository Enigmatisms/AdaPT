"""
    Time domain analysis utilities
    @author: Qianyue He
    @date: 2023-3-20
"""

import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from utils.tools import folder_path

__all__ = ['time_domain_curve']

colors = ("#DF7857", "#4E6E81", "#F99417")

def time_domain_curve(profiles: np.ndarray, window_mode = 'diag_tri', time_step = 1., sol = 1.0, name = "tdom-analysis"):
    # transient profile shape (N, H, W, 3)
    # The intensity is averaged over all components of the spectrum
    # sol: speed of light, 1.0 by default
    transient_num, img_h, img_w, _ = profiles.shape
    max_time = time_step * transient_num / sol
    ts = np.linspace(0., max_time, transient_num)
    if isinstance(window_mode, str):
        if window_mode == 'diag_tri':
            # three window along the image diagonal direction
            win_h, win_w = img_h // 3, img_w // 3
            results = np.zeros((3, transient_num), np.float32)
            intensity = profiles.mean(axis = -1)
            for i in range(3):
                parts = intensity[:, i * win_h:(i + 1) * win_h, i * win_w:(i + 1) * win_w]
                # TODO: to simply model the sensor, we can add weight kernels for averaging step 
                results[i, :] = parts.mean(axis = (-1, -2))      # spatial average 
            for i in range(3):
                plt.scatter(ts, results[i], s = 5, c = colors[i])
                plt.plot(ts, results[i], label = f'diagonal window id = {i+1}', c = colors[i])
            plt.title("AdaPT window temporal analysis (Diagonal cropping)")
        elif window_mode == 'whole':
            results = profiles.mean(axis = (-1, -2, -3))
            plt.scatter(ts, results, s = 5, c = colors[0])
            plt.plot(ts, results, label = f'whole image', c = colors[0])
            plt.title("AdaPT window temporal analysis (whole image)")
        results.astype(np.float32).tofile(f"{folder_path(f'./utils/analysis/')}{name}-{window_mode}.data")
    else:
        raise NotImplementedError("This branch is not urgent, therefore not implemented now.")
    plt.legend()
    plt.grid(axis = 'both')
    plt.xlim((0, max_time))
    plt.xlabel("Temporal progression")
    plt.ylabel("Photon number / Signal Intensity")
    plt.show()

def visualize(data_name: str, time_step = 1., sol = 1.0):
    results = np.fromfile(data_name, np.float32)
    if "diag_tri" in data_name:
        results = results.reshape(3, -1)
        transient_num = results.shape[-1]
        max_time = time_step * transient_num / sol
        ts = np.linspace(0., max_time, transient_num)
        for i in range(3):
            plt.scatter(ts, results[i], s = 5, c = colors[i])
            plt.plot(ts, results[i], label = f'diagonal window id = {i+1}', c = colors[i])
        plt.title("AdaPT window temporal analysis (Diagonal cropping)")
    elif "whole" in data_name:
        transient_num = results.shape[-1]
        max_time = time_step * transient_num / sol
        ts = np.linspace(0., max_time, transient_num)
        plt.scatter(ts, results, s = 5, c = colors[0])
        plt.plot(ts, results, label = f'whole image', c = colors[0])
        plt.title("AdaPT window temporal analysis (whole image)")
    plt.legend()
    plt.grid(axis = 'both')
    plt.xlim((0, max_time))
    plt.xlabel("Temporal progression")
    plt.ylabel("Photon number / Signal Intensity")
    plt.show()

if __name__ == "__main__":
    time_step = 1. if len(sys.argv) < 3 else float(sys.argv[2])
    sol       = 1. if len(sys.argv) < 4 else float(sys.argv[3])
    visualize(sys.argv[1], time_step, sol)