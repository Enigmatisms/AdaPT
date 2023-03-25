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

from scipy import io
from utils.tools import folder_path
from scene.opts import get_tdom_options
from scipy.signal import find_peaks, peak_widths

__all__ = ['time_domain_curve']

colors = ("#DF7857", "#4E6E81", "#F99417")
color_cnt = 0

def lerp(pos, data):
    pos_sid = np.floor(pos).astype(np.int32)
    pos_eid = np.ceil(pos).astype(np.int32)
    return data[pos_sid] * (pos - pos_sid) + data[pos_eid] * (pos_eid - pos)

def peak_analysis(
    curves: np.ndarray, ts: np.ndarray = None, prominence = 0.5, 
    distance = 50, scaler = 1e9, unit = "ns", show = False, 
    fw_cutoff = 5, sub_curve_avg = [0, 2]
):
    # analysis for curves: find peaks and calculate FWHM for each peak
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
    print(f"{len(peaks)} detected, length:")
    for i, width in enumerate(fwhm_width):
        print(f"No.{i+1} peak, width = {width * scaler:.5f} {unit}")
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

def time_domain_curve(profiles: np.ndarray, window_mode = 'diag_tri', time_step = 1., sol = 1.0, name = "tdom-analysis", max_norm = True, viz = True):
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
                # TODO: to simply model the sensor, we can add weight kernels for averaging step 
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
        visualize(results, ts, window_mode, max_time)
    return results, ts

def visualize(results: np.ndarray, ts: np.ndarray, method: str, max_time: float, name = "AdaPT", show = True, whole_legend = 'whole image', extras = None):
    if method == "diag_tri":
        for i in range(3):
            plt.scatter(ts, results[i], s = 4, c = colors[i])
            plt.plot(ts, results[i], label = f'diagonal window id = {i+1}', c = colors[i])
        if show: plt.title(f"{name} window temporal analysis ({whole_legend})")
    elif method == "diag_side_mean":
        if results.ndim > 1:
            results = (results[0] + results[2]) / 2.
        plt.scatter(ts, results, s = 4, c = colors[0])
        plt.plot(ts, results, label = f'side window avg', c = colors[0])
        if show: plt.title(f"{name} window temporal analysis ({whole_legend})")
    elif method == "whole":
        global color_cnt
        if results.ndim > 1:
            results = results.mean(axis = 0)
        plt.scatter(ts, results, s = 5, c = colors[color_cnt])
        plt.plot(ts, results, label = name, c = colors[color_cnt])
        color_cnt += 1
        if show: plt.title(f"{name} window temporal analysis ({whole_legend})")
    if show:
        if extras is not None:
            peaks      = extras['peaks']
            heights    = extras['heights']
            left_ips   = extras['left_ips']
            right_ips  = extras['right_ips']
            start_time = extras['start_time']
            plt.scatter(ts[peaks], results[peaks], s = 40, facecolors = 'none', edgecolors = 'b')
            plt.hlines(heights, left_ips, right_ips, color = "#22BB22", linewidth = 2, label = 'FWHM')
            plt.vlines(start_time, 0., 1.0, color = "#444444", linewidth = 2, linestyles="--", label = f'start time = {start_time}')
        plt.legend()
        plt.grid(axis = 'both')
        plt.xlim((0, max_time))
        plt.xlabel("Temporal progression")
        plt.ylabel("Photon number / Signal Intensity")
        plt.show()

def mat_data_reader(mat_file_path: str, var_name: str, mode = 'diag_tri', time_step = 55e-12, sol = 1.0, viz = False, show = False, whole_legend = 'whole image'):
    """ Matlab file reader 
        Time step second to nanosecond, 55e-12 is the time step of SPAD
    """
    feature_mat: np.ndarray = io.loadmat(mat_file_path)[var_name]
    # the shape of the feature mat is (32, 32, 230), (32, 32) is the spatial resolution of the SPAD
    # 230 means the temporal resolution
    if feature_mat.ndim < 3 and mode != 'whole':
        print(f"[Warning] Mode is set to be '{mode}' but the shape of the feature mat is {feature_mat.shape}, therefore mode is set to 'whole'")
        mode = 'whole'
    if 'diag' in mode:
        img_h, img_w, _ = feature_mat.shape
        results = np.zeros((3, 230))                    # 230 bins for SPAD
        win_h, win_w = img_h // 3, img_w // 3 
        for i in range(3):
            values = feature_mat[i * win_h:(i + 1) * win_h, i * win_w:(i + 1) * win_w, :].mean(axis = (0, 1))
            results[i, :] = values
    else:
        if feature_mat.ndim < 3:               # single channel for one pixel
            results = feature_mat.mean(axis = 0)
        else:                                   # SPAD 32 * 32 * 230
            results = feature_mat.mean(axis = (0, 1))
    results /= results.max()
    transient_num = results.shape[-1]
    max_time = time_step * transient_num / sol
    ts = np.linspace(0., max_time, transient_num)
    if viz:
        visualize(results, ts, mode, max_time, name = "SPAD", show = show, whole_legend = whole_legend)
    return results, ts

def sim_visualize(opts, whole_legend = 'whole image', analyze_peak = True):  
    time_step = opts.sim_interval
    sol       = opts.sim_sol
    file_name = os.path.join(opts.sim_path, opts.sim_name)
    results = np.fromfile(file_name, np.float32)
    if "diag" in file_name:
        results = results.reshape(3, -1)
    results /= results.max()

    transient_num = results.shape[-1]
    max_time = time_step * transient_num / sol
    ts = np.linspace(0., max_time, transient_num)
    extras = None
    if analyze_peak:
        peaks, heights, left_ips, right_ips, start_time = peak_analysis(results, ts)
        extras = {'peaks': peaks, 'heights': heights, 'left_ips': left_ips, 'right_ips': right_ips, 'start_time': start_time}
    visualize(results, ts, opts.window_mode, max_time, whole_legend = f"{whole_legend} of {opts.sim_name}", extras = extras)

if __name__ == "__main__":
    opts = get_tdom_options()
    if opts.mode == 'sim':
        sim_visualize(opts, 'AdaPT simulation')
    else:
        comp_mode = (opts.mode == 'comp')
        file_path = os.path.join(opts.real_path, opts.real_name)
        mat_data_reader(file_path, "transient", mode = opts.window_mode, viz = True, whole_legend = 'real data')
        file_path = os.path.join(opts.theory_path, opts.theory_name)
        mat_data_reader(file_path, "Phi", mode = opts.window_mode, viz = True, show = not comp_mode, whole_legend = 'diffusion theory')
        if comp_mode:
            sim_visualize(opts, 'AdaPT simulation')