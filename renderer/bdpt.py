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

from scene.obj_desc import ObjDescriptor
from renderer.vpt import VolumeRenderer
from renderer.path_utils import Vertex
from renderer.constants import *

vec2i = ttype.vector(2, int)

@ti.data_oriented
class BDPT(VolumeRenderer):
    def __init__(self, emitters: List[LightSource], objects: List[ObjDescriptor], prop: dict):
        super().__init__(emitters, objects, prop)
        
        self.light_paths = Vertex.field()       # (W, H, max_bounce + 1)
        self.cam_paths   = Vertex.field()       # (W, H, max_bounce + 2)

        self.path_nodes = ti.root.dense(ti.ij, (self.w, self.h))
        # light vertex is not included, therefore +1
        self.path_nodes.bitmasked(ti.k, self.max_bounce + 1).place(self.light_paths)       
        # camera vertex and extra light vertex is not included, therefore + 2
        self.path_nodes.bitmasked(ti.k, self.max_bounce + 1).place(self.cam_paths)        
        self.inv_cam_r = self.cam_r.inverse()
        self.cam_normal = (self.cam_r @ vec3([0, 0, 1])).normalized()

        # self.A is the area of the imaging space on z = 1 plane
        self.A = float(self.w * self.h) * (self.inv_focal * self.inv_focal)
        self.max_depth = self.max_bounce
        # TODO: whether the camera is placed inside of an object
        self.free_space_cam = True
        
        # Initial time setting
        self.init_time = 0.      

    @ti.kernel
    def render(self, t_start: int, t_end: int, s_start: int, s_end: int):
        self.cnt[None] += 1
        for i, j in self.pixels:
            cam_vnum = self.generate_eye_path(i, j) + 1
            lit_vnum = self.generate_light_path(i, j) + 1
            # print(f"Processing {i}, {j},  = {cam_vnum}, {lit_vnum}")
            s_end_i = ti.min(lit_vnum, s_end)
            t_end_i = ti.min(cam_vnum, t_end)
            for t in range(t_start, t_end_i):
                for s in range(s_start, s_end_i):
                    depth = s + t - 2
                    if (s == 1 and t == 1) or depth < 0 or depth > self.max_depth:
                        continue
                    multi_light_con = (t > 1) and (s > 0) and (self.cam_paths[i, j, t - 1]._type == VERTEX_EMITTER)
                    if not multi_light_con:
                        radiance, raster_p = self.connect_path(i, j, s, t)
                        if t == 1 and raster_p.min() >= 0:      # non-local contribution
                            ri, rj = raster_p
                            self.color[ri, rj] += ti.select(ti.math.isnan(radiance) | ti.math.isinf(radiance), 0., radiance)      # this op should be atomic
                        else:                                   # local contribution
                            self.color[i, j] += ti.select(ti.math.isnan(radiance) | ti.math.isinf(radiance), 0., radiance)
            self.pixels[i, j] = self.color[i, j] / self.cnt[None]
    
    def reset(self):
        """ Resetting path vertex container """
        self.path_nodes.deactivate_all()

    @ti.func
    def generate_eye_path(self, i: int, j: int):
        ray_d = self.pix2ray(i, j)
        dot_ray = tm.dot(ray_d, self.cam_normal)
        _, pdf_dir = self.pdf_camera(dot_ray)
        # Starting vertex assignment
        self.cam_paths[i, j, 0] = Vertex(_type = VERTEX_CAMERA, obj_id = -1, emit_id = -1, 
            bool_bits = (self.free_space_cam << 1) + 1, time = self.init_time,
            normal = self.cam_normal, pos = self.cam_t, ray_in = ZERO_V3, beta = vec3([1., 1., 1.])
        )
        return self.random_walk(i, j, self.cam_t, ray_d, pdf_dir, ONES_V3, TRANSPORT_RAD) + 1

    @ti.func
    def generate_light_path(self, i: int, j: int):
        # TODO: emitter emitting time is not set
        emitter, emitter_pdf, _ , emit_id = self.sample_light()
        ray_o, ray_d, pdf_pos, pdf_dir, normal = emitter.sample_le(self.precom_vec, self.normals, self.mesh_cnt)
        ret_int = emitter.intensity
        vertex_pdf = pdf_pos * emitter_pdf
        self.light_paths[i, j, 0] = Vertex(_type = VERTEX_EMITTER, obj_id = emitter.obj_ref_id, 
            emit_id = emit_id, bool_bits = emitter.bool_bits, time = 0., pdf_fwd = vertex_pdf, 
            normal = normal, pos = ray_o, ray_in = ZERO_V3, beta = ret_int
        )
        vertex_num = 0
        if pdf_dir > 0. and ret_int.max() > 0. and vertex_pdf > 0.:      # black emitter / inpossible direction 
            beta = ret_int * ti.abs(tm.dot(ray_d, normal)) / (vertex_pdf * pdf_dir)
            # Why we need to put beta in here?
            vertex_num = self.random_walk(i, j, ray_o, ray_d, pdf_dir, beta, TRANSPORT_IMP) + 1
        return vertex_num

    @ti.func
    def random_walk(self, i: int, j: int, init_ray_o, init_ray_d, pdf: float, beta, transport_mode: int):
        """ Random walk to generate path 
            pdf: initial pdf for this path
            transport mode: whether it is radiance or importance, 0 is camera radiance, 1 is light importance
            Before the random walk, corresponding initial vertex should be appended already
            TODO: Extensive logic check and debug should be done here.
            can not reassign function parameter (non-scalar): https://github.com/taichi-dev/taichi/pull/3607
        """
        ray_o      = init_ray_o
        ray_d      = init_ray_d
        throughput = beta
        vertex_num = 0
        acc_time   = 0.         # accumulated time
        ray_pdf    = pdf        # PDF is of solid angle measure, therefore should be converted

        while True:
            # Step 1: ray intersection
            obj_id, normal, min_depth = self.ray_intersect(ray_d, ray_o)

            if obj_id < 0: break    # nothing is hit, break
            vertex_num += 1
            in_free_space = tm.dot(normal, ray_d) < 0
            # Step 2: check for mean free path sampling
            # Calculate mfp, path_beta = transmittance / PDF
            is_mi, min_depth, path_beta = self.sample_mfp(obj_id, in_free_space, min_depth) 
            diff_vec = ray_d * min_depth
            hit_point = diff_vec + ray_o
            hit_light = -1 if is_mi else self.emitter_id[obj_id]
            acc_time += min_depth
            throughput *= path_beta         # attenuate first

            # Do not place vertex on null surface (no correct answer about whether it's surface or medium)
            if not is_mi and not self.non_null_surface(obj_id):    # surface interaction for null surface should be skipped   
                vertex_num -= 1
                ray_o = hit_point
                continue

            # Step 3: Create a new vertex and calculate pdf_fwd
            pdf_fwd = BDPT.convert_density(ray_pdf, diff_vec, normal, is_mi)
            is_delta = (not is_mi) and self.is_delta(obj_id)
            bool_bits = BDPT.get_bool_bits(is_delta, in_free_space, hit_light >= 0)
            vertex_args = {"_type": ti.select(is_mi, VERTEX_MEDIUM, VERTEX_SURFACE), "obj_id": obj_id, "emit_id": hit_light, 
                "bool_bits": bool_bits, "pdf_fwd": pdf_fwd, "time": acc_time, "pos": hit_point,
                "normal": ti.select(is_mi, ZERO_V3, normal), "ray_in": ray_d, "beta": throughput                
            }
            if transport_mode == TRANSPORT_IMP:         # Camera path
                self.light_paths[i, j, vertex_num] = Vertex(**vertex_args) 
            else:                          # Light path
                self.cam_paths[i, j, vertex_num] = Vertex(**vertex_args) 

            # Step 4: ray termination test - RR termination and max bounce. If ray terminates, we won't have to sample
            if vertex_num >= self.max_bounce:
                break
            if throughput.max() < 1e-4: break
            prev_vid = vertex_num - 1

            # Step 5: sample new ray. This should distinguish between surface and medium interactions
            old_ray_d = ray_d

            ray_d, indirect_spec, ray_pdf = self.sample_new_ray(obj_id, old_ray_d, normal, is_mi, in_free_space, transport_mode)
            ray_o = hit_point
            if not is_mi:
                throughput *= (indirect_spec / ray_pdf)

            # Step 6: re-evaluate backward PDF
            pdf_bwd = ray_pdf
            if is_delta:
                ray_pdf = 0.0
                pdf_bwd = 0.
            elif not is_mi: # If current sampling exists on the surface
                pdf_bwd = self.surface_pdf(obj_id, -old_ray_d, normal, -ray_d)
            # ray_o is the position of the current vertex, which is used in prev vertex pdf_bwd
            if transport_mode == TRANSPORT_IMP:         # Camera transport mode
                self.light_paths[i, j, prev_vid].set_pdf_bwd(pdf_bwd, ray_o)
            else:
                self.cam_paths[i, j, prev_vid].set_pdf_bwd(pdf_bwd, ray_o)
        return vertex_num
    
    @ti.func
    def connect_path(self, i: int, j: int, sid: int, tid: int):
        """ Rigorous logic check, review and debug should be done """
        le = ZERO_V3
        sampled_v = Vertex(_type = VERTEX_NULL)         # a default vertex
        vertex_sampled = False                          # whether any new vertex is sampled
        raster_p = vec2i([-1, -1])                      # reprojection for light path - camera direct connection
        if sid == 0:                                    # light path is not used  
            vertex = self.cam_paths[i, j, tid - 1]
            if vertex.is_light():                       # is the current vertex an emitter vertex?
                le = self.src_field[int(vertex.emit_id)].eval_le(vertex.ray_in, vertex.normal) * vertex.beta
        elif tid == 1:                                  # re-rasterize point onto the film, atomic add is allowed
            vertex = self.light_paths[i, j, sid - 1]
            if vertex.is_connectible():
                ray_d = self.cam_t - vertex.pos
                depth = ray_d.norm()
                ray_d /= depth
                in_free_space = vertex.is_in_free_space()
                we, cam_pdf, raster_p = self.sample_camera(ray_d, depth)
                tr2cam = self.track_ray(ray_d, vertex.pos, depth)        # calculate transmittance from vertex to camera
                # Note that `get_pdf` and `eval` are direction dependent
                # camera importance is valid / visible / radiance transferable
                if cam_pdf > 0. and tr2cam.max() > 0:
                    fr2cam = self.eval(int(vertex.obj_id), vertex.ray_in, ray_d, vertex.normal, vertex.is_mi(), in_free_space, TRANSPORT_IMP)
                    bool_bits = self.get_bool_bits(True, self.free_space_cam, False)
                    sampled_v = Vertex(_type = VERTEX_CAMERA, obj_id = -1, emit_id = -1, 
                        bool_bits = bool_bits, time = vertex.time + depth, 
                        normal = self.cam_normal, pos = self.cam_t, ray_in = ray_d, beta = we / cam_pdf
                    )
                    vertex_sampled = True
                    le = vertex.beta * tr2cam * fr2cam * sampled_v.beta
        elif sid == 1:          # only one light vertex is used, resample
            vertex = self.cam_paths[i, j, tid - 1]
            if vertex.is_connectible():
                # randomly sample an emitter and corresponding point (direct component)
                emitter, emitter_pdf, _ev, emit_id = self.sample_light()
                emit_pos, emit_int, _, normal = emitter.         \
                    sample_hit(self.precom_vec, self.normals, self.mesh_cnt, vertex.pos)        # sample light
                to_emitter    = emit_pos - vertex.pos
                emitter_d     = to_emitter.norm()
                to_emitter    = to_emitter / emitter_d
                in_free_space = vertex.is_in_free_space()
                tr2light      = self.track_ray(to_emitter, vertex.pos, emitter_d)   # calculate transmittance from vertex to camera
                # emitter should have non-zero emission / visible / transferable
                if emit_int.max() > 0 and tr2light.max() > 0:
                    fr2light    = self.eval(int(vertex.obj_id), vertex.ray_in, to_emitter, vertex.normal, vertex.is_mi(), in_free_space)
                    sampled_v   = Vertex(_type = VERTEX_EMITTER, obj_id = self.get_associated_obj(int(vertex.emit_id)), 
                        emit_id = emit_id, bool_bits = emitter.bool_bits, time = vertex.time + emitter_d,
                        normal  = normal, pos = emit_pos, ray_in = to_emitter, beta = emit_int / emitter_pdf
                    )
                    vertex_sampled = True
                    sampled_v.pdf_fwd = emitter.area_pdf() / float(self.src_num)
                    le = vertex.beta * tr2light * fr2light * sampled_v.beta
                        # No need to apply cosine term, since fr2cam already includes it
        else:                   # general cases
            cam_v = self.cam_paths[i, j, tid - 1]
            lit_v = self.light_paths[i, j, sid - 1]
            if cam_v.is_connectible() and lit_v.is_connectible():
                cam2lit_v  = lit_v.pos - cam_v.pos
                length     = cam2lit_v.norm()
                cam2lit_v /= length
                cam_in_fspace = cam_v.is_in_free_space()
                lit_in_fspace = lit_v.is_in_free_space()
                tr_con = self.track_ray(cam2lit_v, cam_v.pos, length)   # calculate transmittance from vertex to camera
                if tr_con.max() > 0. and length > 0.:           # if not occluded
                    fr_cam = self.eval(int(cam_v.obj_id), cam_v.ray_in, cam2lit_v, cam_v.normal, cam_v.is_mi(), cam_in_fspace, TRANSPORT_RAD)
                    fr_lit = self.eval(int(lit_v.obj_id), lit_v.ray_in, -cam2lit_v, lit_v.normal, lit_v.is_mi(), lit_in_fspace, TRANSPORT_IMP)
                    # Geometry term: two cosine is in fr_xxx, length^{-2} is directly computed here
                    le = cam_v.beta * fr_cam * (tr_con / (length * length)) * fr_lit * lit_v.beta
        weight = 0.
        if le.max() > 0.:             # zero-contribution will not have MIS weight
            weight = 1.0
            if sid + tid != 2:      # for path with only two vertices, forward and backward is the same
                weight = self.bdpt_mis_weight(sampled_v, vertex_sampled, i, j, sid, tid)
        return le * weight, raster_p
    
    @ti.func
    def update_endpoint(self, cam_end: ti.template(), lit_end: ti.template(), i: int, j: int, idx_t: int, idx_s: int):
        # s + t > 2, since s + t == 2 will not enter `mis_weight`, and s + t < 2 will not have path connection
        if idx_s >= 0:                  # If lit_end is not null vertex
            self.pdf(lit_end, Vertex(_type = VERTEX_NULL), cam_end)
            self.pdf(cam_end, Vertex(_type = VERTEX_NULL), lit_end)
            if idx_t >= 1:
                self.pdf(cam_end, lit_end, self.cam_paths[i, j, idx_t - 1])
            if idx_s >= 1:
                self.pdf(lit_end, cam_end, self.light_paths[i, j, idx_s - 1])
        else:           # idx_t must >= 2       
            # if the camera hits an emitter
            self.pdf_light_origin(cam_end)
            if idx_t >= 1:
                self.pdf_light(cam_end, self.cam_paths[i, j, idx_t - 1])

    @ti.func
    def bdpt_mis_weight(self, sampled_v: ti.template(), valid_sample: int, i: int, j: int, sid: int, tid: int):
        """ Extensive logic check and debugging should be done 
            This is definitely the most complex part, logically
        """
        t_sampled = valid_sample & (tid == 1)
        s_sampled = valid_sample & (sid == 1)
        sum_ri = 0.
        backup = vec4([-1, -1, -1, -1])             # p(t-1), p(t-2), q(s-1), q(s-2)

        backup[0] = self.cam_paths[i, j, tid - 1].pdf_bwd
        if tid > 1:
            backup[1] = self.cam_paths[i, j, tid - 2].pdf_bwd
        if sid > 0:
            backup[2] = self.light_paths[i, j, sid - 1].pdf_bwd
            if sid > 1:
                backup[3] = self.light_paths[i, j, sid - 2].pdf_bwd
        idx_t = tid - 1
        idx_s = sid - 1

        if t_sampled:       # tid == 1, therefore sid > 1 (tid + sid >= 2 and tid sid can't both be 1)
            self.update_endpoint(sampled_v, self.light_paths[i, j, idx_s], i, j, idx_t, idx_s)
        elif s_sampled:
            self.update_endpoint(self.cam_paths[i, j, idx_t], sampled_v, i, j, idx_t, idx_s)
        else:
            if sid == 0:
                self.update_endpoint(self.cam_paths[i, j, idx_t], Vertex(_type = VERTEX_NULL), i, j, idx_t, idx_s)
            else:           # Here idx_s can not be 0 (sid == 1 will go to s_sampled logic above)
                self.update_endpoint(self.cam_paths[i, j, idx_t], self.light_paths[i, j, idx_s], i, j, idx_t, idx_s)

        ri = ti.select(t_sampled, sampled_v.pdf_ratio(), self.cam_paths[i, j, idx_t].pdf_ratio())
        # Avoid indexing one vertex of cam_paths / light_paths twice 
        not_delta = False
        if idx_t > 0 and self.cam_paths[i, j, idx_t - 1].is_connectible():
            not_delta = True
            sum_ri += ri
        while idx_t > 1:
            idx_t -= 1
            ri *= self.cam_paths[i, j, idx_t].pdf_ratio()
            next_not_delta = self.cam_paths[i, j, idx_t - 1].is_connectible()
            if not_delta and next_not_delta:
                sum_ri += ri
            not_delta = next_not_delta
        if idx_s >= 0:                        # sid can be 0, 
            ri = ti.select(s_sampled, sampled_v.pdf_ratio(), self.light_paths[i, j, idx_s].pdf_ratio())
            not_delta = False
            if idx_s >= 0 and self.light_paths[i, j, ti.max(idx_s - 1, 0)].is_connectible():
                not_delta = True
                sum_ri += ri
            while idx_s >= 1:
                idx_s -= 1
                ri *= self.light_paths[i, j, idx_s].pdf_ratio()
                next_not_delta = self.light_paths[i, j, ti.max(idx_s - 1, 0)].is_connectible()
                if not_delta and next_not_delta:
                    sum_ri += ri
                not_delta = next_not_delta

        # Recover from the backup values
        for idx in ti.static(range(2)):
            if tid - 1 - idx >= 0: self.cam_paths[i, j, tid - 1 - idx].pdf_bwd = backup[idx]
            if sid - 1 - idx >= 0: self.light_paths[i, j, sid - 1 - idx].pdf_bwd = backup[idx + 2]

        return 1. / (1. + sum_ri)

    @ti.func
    def rasterize_pinhole(self, local_ray_x: float, local_ray_y: float):
        """ For path with only one camera vertex, ray should be re-rasterized to the film
            ray_d is pointing into the camera, therefore should be negated
        """
        valid_raster = False
        raster_p = vec2i([-1, -1])

        pi = int(self.half_w + 0.5 - local_ray_x / self.inv_focal)
        pj = int(self.half_h + 0.5 + local_ray_y / self.inv_focal)
        if pi >= 0 and pj >= 0 and pi < self.w and pj < self.h:
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
    def pdf(self, cur: ti.template(), prev: ti.template(), next: ti.template()):
        """ Renderer passed in is a reference to BDPT class
            When connect to a new path, end point bwd pdf should be updated
            PDF is used when all three points are presented, next_v is directly modified (no race condition)
            Note that when calling `pdf`, self can never be VERTEX_CAMERA (you can check the logic)
        """
        pdf_sa = 0.
        ray_out = next.pos -  cur.pos
        if cur._type == VERTEX_EMITTER:
            self.pdf_light(cur, next)
        elif cur._type == VERTEX_CAMERA:
            ray_norm = ray_out.norm()
            if ray_norm > 0:
                _pp, pdf_sa = self.pdf_camera(ti.abs(tm.dot(self.cam_normal, ray_out / ray_norm)))
        else:
            is_in_fspace = cur.is_in_free_space()

            ray_in = ti.select(prev._type == VERTEX_NULL, cur.ray_in, cur.pos - prev.pos)
            ray_in_norm = ray_in.norm()
            ray_out_norm = ray_out.norm()
            if ray_in_norm > 0. and ray_out_norm > 0.:
                ray_in /= ray_in_norm
                normed_ray_out = ray_out / ray_out_norm
                # TODO: emitter can be inside the Medium, therefore `is_mi` can not only be `cur._type == VERTEX_MEDIUM`
                pdf_sa = self.get_pdf(int(cur.obj_id), ray_in, normed_ray_out, cur.normal, cur._type == VERTEX_MEDIUM, is_in_fspace)
        if cur._type != VERTEX_EMITTER:
            # convert to area measure for the next node
            next.pdf_bwd = cur.convert_density(next, pdf_sa, ray_out)

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
        prev.pdf_bwd = pdf
    
    @ti.func
    def pdf_light_origin(self, cur: ti.template()):
        """ Calculate density if the current vertex is an emitter vertex """
        cur.pdf_bwd = self.src_field[int(cur.emit_id)].area_pdf() / float(self.src_num)     # uniform emitter selection
    
    @staticmethod
    @ti.func
    def convert_density(pdf: float, diff_vec: vec3, next_nv: vec3, next_mi:int):
        """ Convert solid angle density to unit area density
            next_nv, next_pos, next_mi: normal vector / position / is_mi for the next vertex
        """
        inv_norm2 = 1. / diff_vec.norm_sqr()
        pdf *= inv_norm2
        if not next_mi:
            pdf *= ti.abs(tm.dot(next_nv, diff_vec * ti.sqrt(inv_norm2)))
        return pdf
    
    @staticmethod
    @ti.func
    def get_bool_bits(delta: int, in_fspace: int, is_light: int = False) -> ti.u8:
        return delta | (in_fspace << 1) | (is_light << 2)
    
    def summary(self):
        print(f"[INFO] BDPT Finished rendering. SPP = {self.cnt[None]}. Rendering time: {self.clock.toc():.3f} s")
    