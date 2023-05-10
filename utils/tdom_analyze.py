"""
    Time domain analysis utilities
    @author: Qianyue He
    @date: 2023-3-20
"""

import os
import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt

from utils.tools import folder_path
from parsers.opts import get_tdom_options
from scipy.signal import find_peaks, peak_widths

from rich.console import Console
CONSOLE = Console(width = 128)

colors = ("#DF7857", "#4E6E81", "#F99417")
color_cnt = 0

def lerp(pos, data):
    pos_sid = np.floor(pos).astype(np.int32)
    pos_eid = np.ceil(pos).astype(np.int32)
    return data[pos_sid] * (pos - pos_sid) + data[pos_eid] * (pos_eid - pos)

def peak_analysis(
    curves: np.ndarray, ts: np.ndarray = None, prominence = 0.02, 
    distance = 50, scaler = 1e9, unit = "ns", show = False, 
    fw_cutoff = 5, sub_curve_avg = [0, 2]
):
    # analysis for curves: find peaks and calculate Width for each peak
    if curves.ndim > 1:
        result = np.zeros(curves.shape[-1])
        for index in sub_curve_avg:
            result += curves[index]
        result /= len(sub_curve_avg)
    else:
        result = curves.copy()
    peaks, _ = find_peaks(result, prominence = prominence, distance = distance)
    # currently, the peak FWHF is calculated relatively (not absolutely)
    # for curves that might be affected by ballistic photons, this FWHF method is not accurate
    _, heights, left_ips, right_ips = peak_widths(result, peaks, rel_height = 1 - 1 / np.e)
    if fw_cutoff:
        result_heights = []
        result_lips    = []
        result_rips    = []
        result_peaks   = []
        for peak, h, lip, rip in zip(peaks, heights, left_ips, right_ips):
            if rip - lip > fw_cutoff:
                result_peaks.append(peak)
                result_heights.append(h)
                result_lips.append(lip)
                result_rips.append(rip)
        peaks = np.array(result_peaks, dtype = peaks.dtype)
        heights = np.float32(result_heights)
        left_ips = np.float32(result_lips)
        right_ips = np.float32(result_rips)

    _, s_heights, start_time, _ = peak_widths(result, peaks, rel_height = 0.999)
    # You are going to need linear interpolation then
    if ts is not None:
        left_ips, right_ips = lerp(left_ips, ts), lerp(right_ips, ts)
        start_time = lerp(start_time, ts)
    fwhm_width = right_ips - left_ips
    CONSOLE.log(f"{len(peaks)} detected, length:")
    for i, width in enumerate(fwhm_width):
        CONSOLE.log(f"No.{i+1} peak, width = {width * scaler:.5f} {unit}")
    if show == True:
        ts = np.arange(result.shape[-1]) if ts is None else ts
        plt.plot(ts, result, color = '#FF5533')
        plt.scatter(ts, result, color = '#FF5533', s = 4)
        plt.scatter(ts[peaks], result[peaks], s = 40, facecolors='none', edgecolors='b')
        plt.scatter(start_time, s_heights, s = 40, facecolors='none', edgecolors='#00AA22')
        plt.grid(axis = 'both')
        plt.title(f"Peak number: {len(peaks)}")
        plt.xlabel(f"temporal progression, unit ({unit})")
        plt.hlines(heights, left_ips, right_ips, color="#22BB22", linewidth = 2)
        plt.show()
    return peaks, heights, left_ips, right_ips, start_time

def get_peak_analysis(results:np.ndarray, ts: np.ndarray, opts):
    if opts.analyze_peak:
        peaks, heights, left_ips, right_ips, start_time = peak_analysis(results, ts, prominence = opts.prominence)
        return {'peaks': peaks, 'heights': heights, 'left_ips': left_ips, 'right_ips': right_ips, 'start_time': start_time}
    return None

def time_domain_curve(profiles: np.ndarray, window_mode = 'diag_tri', time_step = 1., sol = 1.0, name = "tdom-analysis", max_norm = False, viz = True):
    # transient profile shape (N, H, W, 3)
    # The intensity is averaged over all components of the spectrum
    # sol: speed of light, 1.0 by default
    transient_num, img_h, img_w, _ = profiles.shape
    if isinstance(window_mode, str):
        if 'diag' in window_mode:
            # three window along the image diagonal direction
            win_h, win_w = img_h // 3, img_w // 3
            results = np.zeros((3, transient_num), np.float32)
            intensity = profiles.mean(axis = -1)
            for i in range(3):
                parts = intensity[:, i * win_h:(i + 1) * win_h, i * win_w:(i + 1) * win_w]
                # TODO: to simply model the sensor. We can of course add weight kernels for averaging step 
                results[i, :] = parts.mean(axis = (-1, -2))      # spatial average 
        elif window_mode == 'whole':
            results = profiles.mean(axis = (-1, -2, -3))
        results.astype(np.float32).tofile(f"{folder_path(f'./utils/analysis/')}{name}-{window_mode}.data")
    else:
        raise NotImplementedError("This branch is not urgent, therefore not implemented now.")
    if max_norm:
        results /= results.max()

    transient_num = results.shape[-1]
    max_time = time_step * transient_num / sol
    ts = np.linspace(0., max_time, transient_num)
    if viz:
        visualizer = Visualizer(window_mode, max_time)
        visualizer.visualize(results, ts)
    return results, ts

class Visualizer:
    color_cnt = 0
    def __init__(self, method: str, max_time: str, name: str = "AdaPT") -> None:
        self.method = method
        self.max_time = max_time
        self.name = name

    def visualize(self, results: np.ndarray, ts: np.ndarray, legend: str = "radiance", show = True, display = True, extras = None):
        if self.method == "diag_tri":
            for i in range(3):
                plt.scatter(ts, results[i], s = 4, c = colors[i])
                plt.plot(ts, results[i], label = f'diagonal window id = {i+1}', c = colors[i])
        elif self.method == "diag_side_mean":
            if results.ndim > 1:
                results = (results[0] + results[2]) / 2.
            plt.scatter(ts, results, s = 4, c = colors[Visualizer.color_cnt])
            plt.plot(ts, results, label = legend, c = colors[Visualizer.color_cnt])
            Visualizer.color_cnt += 1
        elif self.method == "whole":
            if results.ndim > 1:
                results = results.mean(axis = 0)
            plt.scatter(ts, results, s = 5, c = colors[Visualizer.color_cnt])
            plt.plot(ts, results, label = legend, c = colors[Visualizer.color_cnt])
            Visualizer.color_cnt += 1
        if show:
            if extras is not None:
                peaks      = extras['peaks']
                heights    = extras['heights']
                left_ips   = extras['left_ips']
                right_ips  = extras['right_ips']
                start_time = extras['start_time']
                peak_ys = results[peaks] if results.ndim == 1 else results.mean(axis = 0)[peaks]
                plt.scatter(ts[peaks], peak_ys, s = 40, facecolors = 'none', edgecolors = 'b')
                plt.hlines(heights, left_ips, right_ips, color = "#22BB22", linewidth = 2, label = f'Width: {right_ips - left_ips}')
                plt.vlines(start_time, 0., 1.0, color = "#444444", linewidth = 2, linestyles="--", label = f'start time: {start_time}')
            plt.title(f"{self.name} window temporal analysis\n({self.name})")
            plt.legend()
            plt.grid(axis = 'both')
            plt.xlim((0, self.max_time))
            plt.xlabel("Temporal progression")
            plt.ylabel("Photon number / Signal Intensity")
            if display: plt.show()

def sim_visualize(opts, legend = 'radiance'):  
    time_step = opts.sim_interval
    sol       = opts.sim_sol
    file_name = os.path.join(opts.sim_path, opts.sim_name)
    results = np.fromfile(file_name, np.float32)
    if "diag" in file_name:
        results = results.reshape(3, -1)
    if opts.window_mode in {'diag_tri', 'whole'}:
        results /= results.max()
    else:
        results /= (0.5 * results[0] + 0.5 * results[2]).max()

    transient_num = results.shape[-1]
    max_time = time_step * transient_num / sol
    CONSOLE.log(f"Time step: {time_step}, transient num: {transient_num}, max time: {max_time * sol}, sol: {sol}")
    ts = np.linspace(0., max_time, transient_num)
    extras = get_peak_analysis(results, ts, opts)
    viz = Visualizer(opts.window_mode, max_time, name = f"AdaPT {opts.sim_name}")
    viz.visualize(results, ts, legend = legend, display = not opts.save_fig, extras = extras)

if __name__ == "__main__":
    opts = get_tdom_options()
    sim_visualize(opts, 'AdaPT simulation')
    if opts.save_fig:
        plt.savefig(os.path.join(opts.output_folder, f"{opts.sim_name[:-5]}.png"))