"""
    Bidirectional Path Tracer
    Bidirectional volumetric path tracing method 
    @author: Qianyue He
    @date: 2023-2-20
"""

import taichi as ti
import taichi.math as tm
import taichi.types as ttype
from taichi.math import vec3, vec4

from typing import List
from la.cam_transform import *
from emitters.abtract_source import LightSource

from math import log2
from scene.obj_desc import ObjDescriptor
from renderer.vpt import VolumeRenderer
from renderer.path_utils import Vertex, remap_pdf
from renderer.constants import *

vec2i = ttype.vector(2, int)

N_MAX_BOUNCE = 32
T_MAX_BOUNCE = 255
MAX_SAMPLE_CNT = 512

def block_size(val, max_val = 512):
    if val > max_val or val == 0: return max_val
    if val % 32 == 0 or (val & (val-1) == 0): return val
    cand1 = val - val % 32
    cand2 = 1 << int(log2(val))
    return cand1 if cand1 >= cand2 else cand2

@ti.data_oriented
class BDPT(VolumeRenderer):
    def __init__(self, emitters: List[LightSource], objects: List[ObjDescriptor], prop: dict):
        super().__init__(emitters, objects, prop)
        decomp_mode = {'transient_cam': TRANSIENT_CAM, 'transient_lit': TRANSIENT_LIT, 'none': STEADY_STATE}
        decomp_state = decomp_mode[prop.get('decomposition', 'none')]
        sample_cnt       = prop.get('sample_count', 1) if decomp_state else 1
        
        self.light_paths = Vertex.field()
        self.cam_paths   = Vertex.field()
        self.crop_range  = ti.field(ti.i8)              # dummy field, useless when there is no cropping
        self.time_cnts   = ti.field(ti.i32)
        self.time_bins   = ti.Vector.field(3, float)

        self.vertex_cnts = ti.field(ti.i32)

        if self.do_crop:
            # Memory conserving implementation
            self.path_nodes = ti.root.dense(ti.ij, (self.crop_rx << 1, self.crop_ry << 1))
            max_bounce_used = min(T_MAX_BOUNCE, self.max_bounce)
            print(f"[INFO] To conserve memory, dense field will have the same size as the cropped image. ")
            print(f"[INFO] Max bounce allocated: {max_bounce_used}. Small cropped image is recommended.")
            print(f"[INFO] Typically, GPUs will be used, but dynamic memory allocation is not supported in Taichi")
            print(f"[INFO] Therefore, a maximum bounce limit is set here. It can be modified should you wish to, but be careful.")
        else:
            max_bounce_used = N_MAX_BOUNCE
            self.path_nodes = ti.root.dense(ti.ij, (self.w, self.h))
            print(f"[INFO] Max bounce allocated: {N_MAX_BOUNCE}.")
        offsets = (self.start_x, self.start_y, 0)
        self.path_nodes.dense(ti.k, sample_cnt).place(self.time_bins, self.time_cnts, offset = offsets)
        """ Trying opt for dense to leverage BLS: 
            During path connection and vertex construction, global memory ops will consume much time
            if we can do the job in shared memory then it would be better
            An coarse estimate of the shared memory usage: 256 * 64 = 16384B, which should be enough
        """
        self.cam_bitmask = self.path_nodes.dense(ti.k, max_bounce_used + 1)
        self.lit_bitmask = self.path_nodes.dense(ti.k, max_bounce_used + 1)
        self.cam_bitmask.place(self.cam_paths, offset = offsets)    
        self.lit_bitmask.place(self.light_paths, offset = offsets)  
        self.path_nodes.dense(ti.k, 2).place(self.vertex_cnts, offset = offsets)
        self.path_nodes.place(self.crop_range, offset = (offsets[:2]))
        self.conn_path_block_size = block_size(self.max_bounce + 1, 256)

        # ti.profiler.memory_profiler.print_memory_profiler_info()

        if self.max_bounce > max_bounce_used:
            print(f"[Warning] BDPT currently supports only upto {max_bounce_used} bounces per path (either eye or emitter).")

        # camera vertex and extra light vertex is not included, therefore + 2
        self.inv_cam_r = self.cam_r.inverse()
        self.cam_normal = (self.cam_r @ vec3([0, 0, 1])).normalized()

        # For transient rendering: if decomp is none, then it is steady state rendering, otherwise if "transient", then it is transient state
        self.decomp     = ti.field(int, shape = ())
        self.min_time   = ti.field(float, shape = ())
        self.max_time   = ti.field(float, shape = ())
        self.interval   = ti.field(float, shape = ())

        if decomp_state > STEADY_STATE and "interval" not in prop:
            print("[Warning] some transient attributes not in propeties, fall back to default settings.")
        self.decomp[None]     = decomp_state
        self.min_time[None]   = prop.get('min_time', 0.)                                            # lower bounce for time of recording
        self.interval[None]   = prop.get('interval', 0.1)
        self.max_time[None]   = self.min_time[None] + self.interval[None] * sample_cnt  # precomputed max bound

        if self.decomp[None] >= TRANSIENT_CAM:
            print(f"[INFO] Transient state BDPT rendering, start at: {self.min_time[None]:.4f}, step size: {self.interval[None]:.4f}, bin num: {sample_cnt}")
            print(f"[INFO] Transient {'actual camera recording - TRANSIENT_CAM' if self.decomp[None] == TRANSIENT_CAM else 'emitter only - TRANSIENT_LIT'}")
            if prop['sample_count'] > MAX_SAMPLE_CNT:
                print(f"[Warning] sample cnt = {prop['sample_count']} which is larger than {MAX_SAMPLE_CNT}. Bitmasked node might introduce too much memory consumption.")
            if self.interval[None] <= 0:
                raise ValueError("Transient interval must be positive. Otherwise, meaningful or futile.")
        else:
            print("[INFO] Steady state BDPT rendering")

        # self.A is the area of the imaging space on z = 1 plane
        self.A = float(self.w * self.h) * (self.inv_focal * self.inv_focal)
        # TODO: whether the camera is placed inside of an object
        self.free_space_cam = True
        
        # Initial time setting
        self.init_time = 0.   

    @ti.kernel
    def copy_average(self, time_idx: int):
        for i, j in self.crop_range:
            cnt = self.time_cnts[i, j, time_idx]
            self.pixels[i, j] = ti.select(cnt > 0, self.time_bins[i, j, time_idx] / float(cnt), ZERO_V3)

    @ti.kernel
    def render(self, t_start: int, t_end: int, s_start: int, s_end: int, max_bnc: int, max_depth: int):
        self.cnt[None] += 1
        decomp = self.decomp[None]
        min_time = self.min_time[None]
        max_time = self.max_time[None]
        interval = self.interval[None]

        ti.loop_config(parallelize = 8)
        for i, j, k in self.vertex_cnts:
            in_crop_range = i >= self.start_x and i < self.end_x and j >= self.start_y and j < self.end_y
            if in_crop_range:
                if k == 1:
                    self.vertex_cnts[i, j, 1] = ti.min(self.generate_light_path(i, j, max_bnc) + 1, s_end)     # k == 1
                else:
                    self.vertex_cnts[i, j, 0] = ti.min(self.generate_eye_path(i, j, max_bnc) + 1, t_end)       # k == 0
        
        ti.block_local(self.time_cnts)
        ti.block_local(self.time_bins)
        ti.block_local(self.cam_paths)
        ti.block_local(self.light_paths)
        ti.loop_config(parallelize = 8, block_dim = self.conn_path_block_size)
        for i, j, t in self.cam_paths:
            in_crop_range = i >= self.start_x and i < self.end_x and j >= self.start_y and j < self.end_y
            if in_crop_range:
                t_end_i = self.vertex_cnts[i, j, 0]
                s_end_i = self.vertex_cnts[i, j, 1]
                if t >= t_start and t < t_end_i:
                    for s in range(s_start, s_end_i):
                        depth = s + t - 2
                        if (s == 1 and t == 1) or depth < 0 or depth > max_depth:
                            continue
                        multi_light_con = (t > 1) and (s > 0) and (self.cam_paths[i, j, t - 1]._type == VERTEX_EMITTER)
                        if not multi_light_con:
                            radiance, raster_p, path_time = self.connect_path(i, j, s, t, decomp)
                            color = ti.select(ti.math.isnan(radiance) | ti.math.isinf(radiance), 0., radiance)
                            id_i = i
                            id_j = j
                            if t == 1 and raster_p.min() >= 0:      # non-local contribution
                                id_i, id_j = raster_p               # splat samples can only contribute in cropping range
                            if decomp >= TRANSIENT_CAM and path_time < max_time and path_time > min_time:
                                time_idx = int((path_time - min_time) / interval)
                                self.time_bins[id_i, id_j, time_idx] += color   # time_bins and cnts have offset
                                self.time_cnts[id_i, id_j, time_idx] += 1
                            self.color[id_i, id_j] += color
        for i, j in self.pixels:
            self.pixels[i, j] = self.color[i, j] / self.cnt[None]
    
    def reset(self):
        """ Resetting path vertex container """
        pass

    @ti.func
    def generate_eye_path(self, i: int, j: int, max_bnc: int):
        ray_d = self.pix2ray(i, j)
        dot_ray = tm.dot(ray_d, self.cam_normal)
        _, pdf_dir = self.pdf_camera(dot_ray)
        # Starting vertex assignment, note that camera should be a connectible vertex
        self.cam_paths[i, j, 0] = Vertex(_type = VERTEX_CAMERA, obj_id = -1, emit_id = -1, 
            bool_bits = BDPT.get_bool(p_delta = True, in_fspace = self.free_space_cam), time = self.init_time,
            normal = ZERO_V3, pos = self.cam_t, ray_in = ZERO_V3, beta = vec3([1., 1., 1.])
        )
        return self.random_walk(i, j, max_bnc, self.cam_t, ray_d, pdf_dir, ONES_V3, TRANSPORT_RAD) + 1

    @ti.func
    def generate_light_path(self, i: int, j: int, max_bnc: int):
        emitter, emitter_pdf, _ , emit_id = self.sample_light()
        ray_o, ray_d, pdf_pos, pdf_dir, normal = emitter.sample_le(self.precom_vec, self.normals, self.mesh_cnt)
        ret_int = emitter.intensity
        vertex_pdf = pdf_pos * emitter_pdf
        self.light_paths[i, j, 0] = Vertex(_type = VERTEX_EMITTER, obj_id = emitter.obj_ref_id, 
            emit_id = emit_id, bool_bits = emitter.bool_bits, time = emitter.emit_time, pdf_fwd = vertex_pdf, 
            normal = normal, pos = ray_o, ray_in = ZERO_V3, beta = ret_int
        )
        vertex_num = 0
        if pdf_dir > 0. and ret_int.max() > 0. and vertex_pdf > 0.:      # black emitter / inpossible direction 
            beta = ret_int * ti.abs(tm.dot(ray_d, normal)) / (vertex_pdf * pdf_dir)
            # The start time of the current emission should be accounted for
            vertex_num = self.random_walk(i, j, max_bnc, ray_o, ray_d, pdf_dir, beta, TRANSPORT_IMP, emitter.emit_time) + 1
        return vertex_num

    @ti.func
    def random_walk(self, i: int, j: int, max_bnc: int, init_ray_o, init_ray_d, pdf: float, beta, transport_mode: int, acc_time = 0.0):
        """ Random walk to generate path 
            pdf: initial pdf for this path
            transport mode: whether it is radiance or importance, 0 is camera radiance, 1 is light importance
            Before the random walk, corresponding initial vertex should be appended already
            can not reassign function parameter (non-scalar): https://github.com/taichi-dev/taichi/pull/3607
        """
        last_v_pos = init_ray_o
        ray_o      = init_ray_o
        ray_d      = init_ray_d
        throughput = beta
        vertex_num = 0
        ray_pdf    = pdf                # PDF is of solid angle measure, therefore should be converted
        in_free_space = True

        while True:
            # Step 1: ray intersection
            obj_id, normal, min_depth = self.ray_intersect(ray_d, ray_o)

            if obj_id < 0:     
                if not self.world_scattering: break     # nothing is hit, break
                else:                                   # the world is filled with scattering medium
                    # TODO: This may not be totally correct, should be double-checked
                    min_depth = self.world_bound_time(ray_o, ray_d)
                    in_free_space = True
                    obj_id = -1
            else:
                in_free_space = tm.dot(normal, ray_d) < 0

            # Step 2: check for mean free path sampling
            # Calculate mfp, path_beta = transmittance / PDF
            is_mi, min_depth, path_beta = self.sample_mfp(obj_id, in_free_space, min_depth) 
            if obj_id < 0 and not is_mi: break  # exiting world bound
            throughput *= path_beta             # attenuate first
            if throughput.max() < 5e-5: break
                
            hit_point = ray_d * min_depth + ray_o
            hit_light = -1 if is_mi else self.emitter_id[obj_id]
            acc_time += min_depth * self.get_ior(obj_id, in_free_space)

            # Do not place vertex on null surface (no correct answer about whether it's surface or medium)
            if not is_mi and not self.non_null_surface(obj_id):    # surface interaction for null surface should be skipped   
                ray_o = hit_point
                continue

            # Step 3: Create a new vertex and calculate pdf_fwd
            pdf_fwd = BDPT.convert_density(ray_pdf, hit_point - last_v_pos, normal, is_mi)
            last_v_pos = hit_point
            is_delta = (not is_mi) and self.is_delta(obj_id)
            bool_bits = BDPT.get_bool(d_delta = is_delta, is_area = (hit_light >= 0), in_fspace = in_free_space, is_delta = is_delta)
            
            vertex_args = {"_type": ti.select(is_mi, VERTEX_MEDIUM, VERTEX_SURFACE), "obj_id": obj_id, "emit_id": hit_light, 
                "bool_bits": bool_bits, "pdf_fwd": pdf_fwd, "time": acc_time, "pos": hit_point,
                "normal": ti.select(is_mi, ZERO_V3, normal), "ray_in": ray_d, "beta": throughput                
            }
            vertex_num += 1
            if transport_mode == TRANSPORT_IMP:         # Camera path
                self.light_paths[i, j, vertex_num] = Vertex(**vertex_args) 
            else:                          # Light path
                self.cam_paths[i, j, vertex_num] = Vertex(**vertex_args) 

            # Step 4: ray termination test - RR termination and max bounce. If ray terminates, we won't have to sample
            if vertex_num >= max_bnc:
                break
            # TODO: Different strategy for ray termination
            # For transient imaging, simple RR or other termination strategies are not very good
            prev_vid = vertex_num - 1

            # Step 5: sample new ray. This should distinguish between surface and medium interactions
            old_ray_d = ray_d
            ray_d, indirect_spec, ray_pdf = self.sample_new_ray(obj_id, old_ray_d, normal, is_mi, in_free_space, transport_mode)
            ray_o = hit_point
            pdf_bwd = ray_pdf
            if not is_mi:
                if indirect_spec.max() == 0. or ray_pdf == 0.: break
                throughput *= (indirect_spec / ray_pdf)
                if is_delta:
                    ray_pdf = 0.0
                    pdf_bwd = 0.
                else:
                    # Step 6: re-evaluate backward PDF
                    pdf_bwd = self.surface_pdf(obj_id, -old_ray_d, normal, -ray_d)

            # ray_o is the position of the current vertex, which is used in prev vertex pdf_bwd
            if transport_mode == TRANSPORT_IMP:         # Camera transport mode
                self.light_paths[i, j, prev_vid].set_pdf_bwd(pdf_bwd, ray_o)
            else:
                self.cam_paths[i, j, prev_vid].set_pdf_bwd(pdf_bwd, ray_o)
        return vertex_num
    
    @ti.func
    def connect_path(self, i: int, j: int, sid: int, tid: int, decomp: int):
        le = ZERO_V3
        ret_time = 0.
        sampled_v = Vertex(_type = VERTEX_NULL)         # a default vertex
        vertex_sampled = False                          # whether any new vertex is sampled
        raster_p = vec2i([-1, -1])                      # reprojection for light path - camera direct connection
        calc_transmittance = False                      # whether to calculate transmittance
        depth = -1.
        track_pos = ZERO_V3
        connect_dir = ZERO_V3
        if sid == 0:                                    # light path is not used  
            vertex = self.cam_paths[i, j, tid - 1]
            if vertex.is_light():                       # is the current vertex an emitter vertex?
                le = self.src_field[int(vertex.emit_id)].eval_le(vertex.ray_in, vertex.normal) * vertex.beta
                ret_time = self.src_field[int(vertex.emit_id)].emit_time      # for emission, we should acount for their emission time
                if decomp == TRANSIENT_CAM:             # for emission, we should acount for their emission time
                    ret_time += vertex.time      
        elif tid == 1:                                  # re-rasterize point onto the film, atomic add is allowed
            vertex = self.light_paths[i, j, sid - 1]
            if vertex.is_connectible():
                connect_dir = self.cam_t - vertex.pos
                depth       = connect_dir.norm()
                connect_dir /= depth
                in_free_space = vertex.is_in_free_space()
                we, cam_pdf, raster_p = self.sample_camera(connect_dir, depth)
                track_pos      = vertex.pos
                # camera importance is valid / visible / radiance transferable
                if cam_pdf > 0:
                    fr2cam = self.eval(int(vertex.obj_id), vertex.ray_in, connect_dir, vertex.normal, vertex.is_mi(), in_free_space, TRANSPORT_IMP)
                    bool_bits = BDPT.get_bool(True, in_fspace = self.free_space_cam)
                    sampled_v = Vertex(_type = VERTEX_CAMERA, obj_id = -1, emit_id = -1, 
                        bool_bits = bool_bits, time = vertex.time + depth, 
                        normal = self.cam_normal, pos = self.cam_t, ray_in = ZERO_V3, beta = we / cam_pdf
                    )
                    vertex_sampled = True
                    calc_transmittance = fr2cam.max() > 0
                    le = vertex.beta * fr2cam * sampled_v.beta
                    ret_time = vertex.time
        elif sid == 1:          # only one light vertex is used, resample
            vertex = self.cam_paths[i, j, tid - 1]
            if vertex.is_connectible():
                # randomly sample an emitter and corresponding point (direct component)
                emitter, emitter_pdf, _ev, emit_id = self.sample_light()
                emit_pos, emit_int, _, normal = emitter.         \
                    sample_hit(self.precom_vec, self.normals, self.mesh_cnt, vertex.pos)        # sample light
                connect_dir    = emit_pos - vertex.pos
                depth          = connect_dir.norm()
                connect_dir    = connect_dir / depth
                track_pos      = vertex.pos
                in_free_space = vertex.is_in_free_space()
                # emitter should have non-zero emission / visible / transferable
                if emit_int.max() > 0:
                    fr2light    = self.eval(int(vertex.obj_id), vertex.ray_in, connect_dir, vertex.normal, vertex.is_mi(), in_free_space, TRANSPORT_RAD)
                    sampled_v   = Vertex(_type = VERTEX_EMITTER, obj_id = self.get_associated_obj(emit_id), emit_id = emit_id, 
                        bool_bits = emitter.bool_bits, time = emitter.emit_time, pdf_fwd = emitter.area_pdf() / float(self.src_num),
                        normal  = normal, pos = emit_pos, ray_in = ZERO_V3, beta = emit_int / emitter_pdf
                    )
                    vertex_sampled = True
                    calc_transmittance = fr2light.max() > 0
                    le = vertex.beta * fr2light * sampled_v.beta
                    ret_time = emitter.emit_time
                    if decomp == TRANSIENT_CAM:
                        ret_time += vertex.time
        else:                   # general cases
            cam_v = self.cam_paths[i, j, tid - 1]
            lit_v = self.light_paths[i, j, sid - 1]
            if cam_v.is_connectible() and lit_v.is_connectible():
                connect_dir  = lit_v.pos - cam_v.pos
                depth        = connect_dir.norm()
                connect_dir /= depth
                track_pos    = cam_v.pos
                cam_in_fspace = cam_v.is_in_free_space()
                lit_in_fspace = lit_v.is_in_free_space()
                if depth > 0.:           # if not occluded
                    fr_cam = self.eval(int(cam_v.obj_id), cam_v.ray_in, connect_dir, cam_v.normal, cam_v.is_mi(), cam_in_fspace, TRANSPORT_RAD)
                    fr_lit = self.eval(int(lit_v.obj_id), lit_v.ray_in, -connect_dir, lit_v.normal, lit_v.is_mi(), lit_in_fspace, TRANSPORT_IMP)
                    # Geometry term: two cosine is in fr_xxx, length^{-2} is directly computed here
                    calc_transmittance = fr_cam.max() > 0 and fr_lit.max() > 0
                    le = cam_v.beta * fr_cam * fr_lit * lit_v.beta / (depth * depth)
                    ret_time = lit_v.time
                    if decomp == TRANSIENT_CAM:
                        ret_time += cam_v.time
        if le.max() > 0 and calc_transmittance == True:
            tr, track_depth = self.track_ray(connect_dir, track_pos, depth)
            le *= tr
            if decomp == TRANSIENT_CAM:
                ret_time += track_depth
        weight = 0.
        if ti.static(self.use_mis):
            if le.max() > 0:     # zero-contribution will not have MIS weight, it could be possible that after applying the transmittance, le is 0
                weight = 1.0
                if sid + tid != 2:      # for path with only two vertices, forward and backward is the same
                    weight = self.bdpt_mis_weight(sampled_v, vertex_sampled, i, j, sid, tid)
        else:
            weight = 1.0
        result = le * weight
        if result.max() == 0.: ret_time = 0.
        return result, raster_p, ret_time
    
    @ti.func
    def update_endpoint(self, cam_end: ti.template(), lit_end: ti.template(), ratio: ti.template(), i: int, j: int, idx_t: int, idx_s: int):
        # s + t > 2, since s + t == 2 will not enter `mis_weight`, and s + t < 2 will not have path connection
        if idx_s >= 0:                  # If lit_end is not null vertex
            prev_pos = ti.select(idx_t < 1, ZERO_V3, self.cam_paths[i, j, idx_t - 1].pos)
            ratio[2] = self.pdf_ratio(cam_end, prev_pos, lit_end, idx_t < 1)
            if idx_t >= 1:
                ratio[1] = self.pdf_ratio(cam_end, lit_end.pos, self.cam_paths[i, j, idx_t - 1])
            prev_pos = ti.select(idx_s < 1, ZERO_V3, self.light_paths[i, j, idx_s - 1].pos)
            ratio[0] = self.pdf_ratio(lit_end, prev_pos, cam_end, idx_s < 1)
            if idx_s >= 1:
                ratio[3] = self.pdf_ratio(lit_end, cam_end.pos, self.light_paths[i, j, idx_s - 1])
        else:           # idx_t must >= 2       
            # if the camera hits an emitter
            ratio[0] = remap_pdf(self.src_field[int(cam_end.emit_id)].area_pdf() / float(self.src_num)) / remap_pdf(cam_end.pdf_fwd)
            if idx_t >= 1:
                ratio[1] = remap_pdf(self.pdf_light(cam_end, self.cam_paths[i, j, idx_t - 1])) / remap_pdf(self.cam_paths[i, j, idx_t - 1].pdf_fwd)

    @ti.func
    def bdpt_mis_weight(self, sampled_v, valid_sample: int, i: int, j: int, sid: int, tid: int):
        """ Extensive logic check and debugging should be done 
            This is definitely the most complex part, logically
        """
        t_sampled = valid_sample & (tid == 1)
        s_sampled = valid_sample & (sid == 1)
        sum_ri = 0.

        ratios = vec4([-1, -1, -1, -1])             # p(t-1), p(t-2), q(s-1), q(s-2)
        idx_t = tid - 1
        idx_s = sid - 1

        cam_side = self.cam_paths[i, j, idx_t]
        lit_side = Vertex(_type = VERTEX_NULL)
        if s_sampled:
            lit_side = sampled_v
        else:
            if t_sampled:
                cam_side = sampled_v
            if idx_s >= 0:
                lit_side = self.light_paths[i, j, idx_s]

        self.update_endpoint(cam_side, lit_side, ratios, i, j, idx_t, idx_s)

        ri = ratios[0]
        # Avoid indexing one vertex of cam_paths / light_paths twice 
        not_delta = False
        if idx_t > 0 and self.cam_paths[i, j, idx_t - 1].not_delta():
            not_delta = True
            sum_ri += ri
        while idx_t > 1:
            idx_t -= 1
            if ratios[1] <= 0.:
                ri *= self.cam_paths[i, j, idx_t].pdf_ratio()
            else:
                ri *= ratios[1]
                ratios[1] = -1.
            next_not_delta = self.cam_paths[i, j, idx_t - 1].not_delta()
            if not_delta and next_not_delta:
                sum_ri += ri
            not_delta = next_not_delta
        if idx_s >= 0:                        # sid can be 0, 
            ri = ratios[2]
            not_delta = False
            current_not_delta = self.light_paths[i, j, idx_s - 1].not_delta() if idx_s >= 1 else self.light_paths[i, j, 0].not_delta_source()
            if current_not_delta:
                not_delta = True
                sum_ri += ri
            while idx_s >= 1:
                idx_s -= 1
                if ratios[3] <= 0.:
                    ri *= self.light_paths[i, j, idx_s].pdf_ratio()
                else:
                    ri *= ratios[3]
                    ratios[3] = -1.
                next_not_delta = self.light_paths[i, j, idx_s - 1].not_delta() if idx_s >= 1 else self.light_paths[i, j, 0].not_delta_source()
                if not_delta and next_not_delta:
                    sum_ri += ri
                not_delta = next_not_delta

        return 1. / (1. + sum_ri)

    @ti.func
    def rasterize_pinhole(self, local_ray_x: float, local_ray_y: float):
        """ For path with only one camera vertex, ray should be re-rasterized to the film
            ray_d is pointing into the camera, therefore should be negated
        """
        valid_raster = False
        raster_p = vec2i([-1, -1])

        pi = int(self.half_w + 1.0 - local_ray_x / self.inv_focal)
        pj = int(self.half_h + 1.0 + local_ray_y / self.inv_focal)
        if pi >= self.start_x and pj >= self.start_y and pi < self.end_x and pj < self.end_y:   # cropping is considered
            raster_p = vec2i([pi, pj]) 
            valid_raster = True
        return raster_p, valid_raster

    @ti.func
    def sample_camera(self, ray_d: vec3, depth: float):
        """ Though currently, the cam model is pinhole, we still need to calculate
            - Rasterized pixel pos / PDF (solid angle measure) / visibility
            - returns: we, pdf, camera_normal, rasterized position, 
        """
        pdf = 0.0
        we = ZERO_V3
        raster_p = vec2i([-1, -1])
        dot_normal = -tm.dot(ray_d, self.cam_normal)
        if dot_normal > 0.:
            local_ray = self.inv_cam_r @ (-ray_d)
            z = local_ray[2]
            if z > 0:
                local_ray /= z
                raster_p, is_valid = self.rasterize_pinhole(local_ray[0], local_ray[1])
                if is_valid:        # not valid --- outside of imaging plane
                    # For pinhole camera, lens area is 1., this pdf is already in sa measure 
                    pdf = depth * depth / dot_normal
                    we.fill(1.0 / (self.A * ti.pow(dot_normal, 4)))
        return we, pdf, raster_p
    
    @ti.func
    def pdf_camera(self, dot_normal: float):
        """ PDF camera does not rasterize point, we assume that the point is within view frustum
            Implementation see reference: PBR-Book chapter 16-1, equation 16.2
            https://www.pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/The_Path-Space_Measurement_Equation#eq:importance-sa
        """
        pdf_pos = 1.
        pdf_dir = 1.
        if dot_normal > 0.:     # No need to rasterize again
            pdf_dir /= (self.A * ti.pow(dot_normal, 3))
        else:
            pdf_pos = 0.
            pdf_dir = 0.
        return pdf_pos, pdf_dir
    
    @ti.func
    def pdf_ratio(self, cur: ti.template(), prev_pos, next: ti.template(), prev_null = False):
        """ Renderer passed in is a reference to BDPT class
            When connect to a new path, end point bwd pdf should be updated
            PDF is used when all three points are presented, next_v is directly modified (no race condition)
            Note that when calling `pdf`, self can never be VERTEX_CAMERA (you can check the logic)
        """
        pdf_sa = 0.
        pdf_area = 0.0
        ray_out = next.pos - cur.pos
        if cur._type == VERTEX_EMITTER:
            pdf_area = self.pdf_light(cur, next)
        elif cur._type == VERTEX_CAMERA:
            ray_norm = ray_out.norm()
            if ray_norm > 0:
                _pp, pdf_sa = self.pdf_camera(ti.abs(tm.dot(self.cam_normal, ray_out / ray_norm)))
        else:
            is_in_fspace = cur.is_in_free_space()

            ray_in = ti.select(prev_null, ZERO_V3, (cur.pos - prev_pos).normalized())
            ray_out_norm = ray_out.norm()
            if ray_out_norm > 0.:
                normed_ray_out = ray_out / ray_out_norm
                # FIXME: emitter can be inside the Medium (not in the free space), therefore `is_mi` can not only be `cur._type == VERTEX_MEDIUM`
                pdf_sa = self.get_pdf(int(cur.obj_id), ray_in, normed_ray_out, cur.normal, cur._type == VERTEX_MEDIUM, is_in_fspace)
        if cur._type != VERTEX_EMITTER:
            # convert to area measure for the next node
            pdf_area = next.get_pdf_bwd(pdf_sa, cur.pos)
        return remap_pdf(pdf_area) / remap_pdf(next.pdf_fwd)

    @ti.func
    def pdf_light(self, cur: ti.template(), prev: ti.template()):
        """ Calculate directional density (then convert to area measure) for prev_v.pdf_bwd """
        ray_dir  = prev.pos - cur.pos
        inv_len  = 1. / ray_dir.norm()
        ray_dir *= inv_len
        pdf = self.src_field[int(cur.emit_id)].direction_pdf(ray_dir, cur.normal)
        if prev.has_normal():
            pdf *= ti.abs(tm.dot(ray_dir, prev.normal))
        pdf *= (inv_len * inv_len)
        return pdf
    
    @staticmethod
    @ti.func
    def convert_density(pdf: float, diff_vec: vec3, next_nv: vec3, next_mi:int):
        """ Convert solid angle density to unit area density
            next_nv, next_pos, next_mi: normal vector / position / is_mi for the next vertex
        """
        if pdf > 0.:
            inv_norm2 = 1. / diff_vec.norm_sqr()
            pdf *= inv_norm2
            if not next_mi:
                pdf *= ti.abs(tm.dot(next_nv, diff_vec * ti.sqrt(inv_norm2)))
        return pdf
    
    @staticmethod
    @ti.func
    def get_bool(p_delta = False, d_delta = False, is_area = False, is_inf = False, in_fspace = True, is_delta = False):
        return p_delta + (d_delta << 1) + (is_area << 2) + (is_inf << 3) + (in_fspace << 4) + (is_delta << 5)
    
    def summary(self):
        print(f"[INFO] BDPT Finished rendering. SPP = {self.cnt[None]}. Rendering time: {self.clock.toc():.3f} s")
    