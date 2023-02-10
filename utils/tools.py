"""
    Utility functions
    @author: Qianyue He
    @date: 2023.2.10
"""

import os
from time import time

__all__ = ['TicToc', 'timing', 'show_global_config', 'folder_path']

on_off = lambda x: "[ON]" if x else "[OFF]"

class TicToc:
    def __init__(self) -> None:
        self.tic()

    def tic(self): self.start_t = time()
    def toc(self, to_ms = False): return (time() - self.start_t) * (1. if to_ms == False else 1e3)

def timing(verbose = True):
    """ Timer decorator: verbose -- if False, outputs nothing """
    def outter_wrapper(func):
        def inner_wrapper(*args, **kwargs):
            start_time = time()
            ret_val = func(*args, **kwargs)
            if verbose:
                print(f"[Info] Function <{func.__name__}> takes {(time() - start_time) * 1e3:.4f} ms")
            return ret_val
        return inner_wrapper
    return outter_wrapper

def show_global_config(config: dict):
    print(f"[Info] Image to render: (w, h) = ({config['film']['width']}, \
          {config['film']['height']}) with {config['max_bounce']} max bounces")
    print(f"[Info] FOV: {config['fov']:.4f}°. RR: {on_off(config['use_rr'])}.\
          MIS: {on_off(config['use_mis'])}. Shadow rays: {config['shadow_rays']}")

def folder_path(path: str):
    if not os.path.exists(path):
        os.mkdir(path)
    return path
    