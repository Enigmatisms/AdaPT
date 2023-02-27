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

"""
    TODO for 2.26: 
    1. Implement render function
"""

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
        # TODO: max_depth is different from max_bounce
        self.max_depth = 16
        # TODO: whether the camera is placed inside of an object
        self.free_space_cam = True
        self.null = Vertex(_type = VERTEX_NULL)
        
        # Initial time setting
        self.init_time = 0.      

    @ti.kernel
    def render(self):
        self.cnt[None] += 1
        for i, j in self.pixels:
            cam_vnum = self.generate_eye_path(i, j)
            lit_vnum = self.generate_light_path(i, j)
            for t in range(1, cam_vnum):
                for s in range(0, lit_vnum):
                    depth = s + t - 2
                    if (s == 1 and t == 1) or depth < 0 or depth > self.max_depth:
                        continue
                    radiance, raster_p = self.connect_path(i, j, s, t)
                    if t == 1 and raster_p.min() >= 0:      # non-local contribution
                        ri, rj = raster_p
                        self.color[ri, rj] += radiance      # this op should be atomic
                    else:                                   # local contribution
                        self.color[i, j] += radiance
            # TODO: Synchronization will introduce overhead? (Is ti.sync necessary?)
            self.pixels[i, j] = self.color[i, j] / self.cnt[None]
    
    def reset(self):
        """ Resetting path vertex container """
        self.path_nodes.deactivate_all()

    @ti.func
    def generate_eye_path(self, i: int, j: int):
        ray_d = self.pix2ray(i, j)
        _, pdf_dir = self.pdf_camera(ray_d)
        # Starting vertex assignment
        self.cam_paths[i, j, 0] = Vertex(_type = VERTEX_CAMERA, obj_id = -1, emit_id = -1, 
            bool_bits = (self.free_space_cam << 1) + 1, time = self.init_time,
            normal = self.cam_normal, pos = self.cam_t, ray_in = ZERO_V3, beta = vec3([1., 1., 1.])
        )
        return self.random_walk(i, j, self.cam_t, ray_d, pdf_dir, TRANSPORT_RAD) + 1

    @ti.func
    def generate_light_path(self, i: int, j: int):
        # TODO: emitter emitting time is not set
        emitter, _, _ , emit_id = self.sample_light()
        ret_int, ray_o, ray_d, pdf_dir, normal = emitter.sample_le(self.precom_vec, self.normals, self.mesh_cnt)
        self.light_paths[i, j, 0] = Vertex(_type = VERTEX_EMITTER, obj_id = emitter.obj_ref_id, 
            emit_id = emit_id, bool_bits = emitter.bool_bits, time = 0., normal = normal, 
            pos = ray_o, ray_in = ZERO_V3, beta = ret_int
        )
        vertex_num = 0
        if pdf_dir > 0. or ret_int.max() > 0.:      # black emitter / inpossible direction 
            vertex_num = self.random_walk(i, j, ray_o, ray_d, pdf_dir, TRANSPORT_IMP) + 1
        return vertex_num

    @ti.func
    def random_walk(self, i: int, j: int, ray_o: vec3, ray_d: vec3, pdf: float, transport_mode: int):
        """ Random walk to generate path 
            pdf: initial pdf for this path
            transport mode: whether it is radiance or importance, 0 is camera radiance, 1 is light importance
            Before the random walk, corresponding initial vertex should be appended already
            TODO: Extensive logic check and debug should be done here.
        """
        old_pos    = ray_o
        throughput = ONES_V3
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
            hit_point = ray_d * min_depth + ray_o
            hit_light = -1 if is_mi else self.emitter_id[obj_id]
            acc_time += min_depth
            throughput *= path_beta         # attenuate first

            # Step 3: Create a new vertex and calculate pdf_fwd
            # TODO: logic with old / new updates tend to be erroneous
            pdf_fwd = BDPT.convert_density(ray_pdf, old_pos, normal, hit_point, is_mi)
            is_delta = (not is_mi) and self.is_delta(obj_id)
            vertex_args = {"_type": BDPT.interact_mode(is_mi, hit_light), "obj_id": obj_id, "emit_id": hit_light, 
                "bool_bits": (in_free_space << 1) + is_delta, "pdf_fwd": pdf_fwd, "time": acc_time, 
                "normal": ti.select(is_mi, ZERO_V3, normal), "pos": hit_point, "ray_in": ray_d, "beta": throughput                
            }
            if transport_mode == TRANSPORT_RAD:         # Camera path
                self.cam_paths[i, j, vertex_num] = Vertex(**vertex_args) 
            else:                          # Light path
                self.light_paths[i, j, vertex_num] = Vertex(**vertex_args) 

            # Step 4: ray termination test - RR termination and max bounce. If ray terminates, we won't have to sample
            if vertex_num >= self.max_bounce:
                break
            max_value = throughput.max()
            if ti.random(float) > max_value: break
            else: throughput *= 1. / ti.max(max_value, 1e-7)    # unbiased calculation
            prev_vid = vertex_num - 1

            # Step 5: sample new ray. This should distinguish between surface and medium interactions
            old_ray_d = ray_d

            ray_d, indirect_spec, ray_pdf = self.sample_new_ray(obj_id, old_ray_d, normal, is_mi, in_free_space, transport_mode)
            old_pos = ray_o
            ray_o = hit_point
            throughput *= (indirect_spec / ray_pdf)

            # Step 6: re-evaluate backward PDF
            pdf_bwd = ray_pdf
            if not is_mi:   # If current sampling exists on the surface
                pdf_bwd = self.surface_pdf(obj_id, -old_ray_d, normal, -ray_d)
            if is_delta:
                pdf_fwd = 0.
                pdf_bwd = 0.
            # ray_o is the position of the current vertex, which is used in prev vertex pdf_bwd
            if transport_mode == TRANSPORT_RAD:         # Camera transport mode
                self.cam_paths[i, j, prev_vid].set_pdf_bwd(pdf_bwd, ray_o)
            else:
                self.light_paths[i, j, prev_vid].set_pdf_bwd(pdf_bwd, ray_o)
        return vertex_num
    
    @ti.func
    def connect_path(self, i: int, j: int, sid: int, tid: int):
        """ Rigorous logic check, review and debug should be done 
        """
        le = ZERO_V3
        sampled_v = Vertex()        # a default vertex
        vertex_sampled = False      # whether any new vertex is sampled
        raster_p = vec2i([-1, -1])  # reprojection for light path - camera direct connection
        if sid == 0:                # light path is not used  
            vertex = self.cam_paths[i, j, tid - 1]
            if vertex._type == 2:   # is the current vertex an emitter vertex?
                le = self.src_field[vertex.emit_id].eval_le(vertex.ray_in, vertex.normal)
        elif tid == 1:          # re-rasterize point onto the film, atomic add is allowed
            vertex = self.light_paths[i, j, sid - 1]
            if vertex.is_connectible():
                ray_d = self.cam_t - vertex.pos
                depth = ray_d.norm()
                ray_d /= depth
                in_free_space = vertex.is_in_free_space()
                we, cam_pdf, raster_p = self.sample_camera(ray_d, depth)
                tr2cam = self.track_ray(ray_d, vertex.pos, depth)       # calculate transmittance from vertex to camera

                # Note that `get_pdf` and `eval` are direction dependent
                pdf2cam = self.get_pdf(vertex.obj_id, vertex.ray_in, ray_d, vertex.normal, vertex.v_is_mi(), in_free_space)
                # camera importance is valid / visible / radiance transferable
                if cam_pdf > 0. and tr2cam.max() > 0. and pdf2cam > 0.:
                    fr2cam = self.eval(vertex.obj_id, vertex.ray_in, ray_d, vertex.normal, \
                        vertex.v_is_mi(), in_free_space, TRANSPORT_IMP) / pdf2cam
                    sampled_v = Vertex(_type = VERTEX_CAMERA, obj_id = -1, emit_id = -1, 
                        bool_bits = (self.free_space_cam << 1) + 1, time = vertex.time + depth, 
                        normal = self.cam_normal, pos = self.cam_t, ray_in = ray_d, beta = we / cam_pdf
                    )
                    vertex_sampled = True
                    le = vertex.beta * tr2cam * fr2cam * sampled_v.beta
        elif sid == 1:          # only one light vertex is used, resample
            vertex = self.cam_paths[i, j, tid - 1]
            if vertex.is_connectible():
                # randomly sample an emitter and corresponding point (direct component)
                emitter, emitter_pdf, emitter_valid, emit_id = self.sample_light()
                if emitter_valid:
                    emit_pos, emit_int, direct_pdf, normal = emitter.         \
                        sample_hit(self.precom_vec, self.normals, self.mesh_cnt, vertex.pos)        # sample light
                    light_pdf     = direct_pdf * emitter_pdf
                    to_emitter    = emit_pos - vertex.pos
                    emitter_d     = to_emitter.norm()
                    to_emitter    = to_emitter / emitter_d
                    in_free_space = vertex.is_in_free_space()
                    tr2light      = self.track_ray(to_emitter, vertex.pos, emitter_d)   # calculate transmittance from vertex to camera
                    pdf2light     = self.get_pdf(vertex.obj_id, vertex.ray_in, to_emitter, vertex.normal, vertex.v_is_mi(), in_free_space)
                    # emitter should have non-zero emission / visible / transferable
                    if emit_int.max() > 0. and tr2light.max() > 0. and pdf2light > 0.:
                        fr2cam    = self.eval(vertex.obj_id, vertex.ray_in, to_emitter, vertex.normal, vertex.v_is_mi(), in_free_space) / pdf2light
                        sampled_v = Vertex(_type = VERTEX_EMITTER, obj_id = self.get_associated_obj(vertex.emit_id), 
                            emit_id = emit_id, bool_bits = emitter.bool_bits, time = vertex.time + emitter_d,
                            normal = normal, pos = emit_pos, ray_in = to_emitter, beta = emit_int / light_pdf
                        )
                        vertex_sampled = True
                        sampled_v.pdf_fwd = emitter.area_pdf(to_emitter, normal) 
                        le = vertex.beta * tr2cam * fr2cam * sampled_v.beta
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
                cam_pdf = self.get_pdf(cam_v.obj_id, cam_v.ray_in, cam2lit_v, cam_v.normal, cam_v.v_is_mi(), cam_in_fspace)
                lit_pdf = self.get_pdf(lit_v.obj_id, lit_v.ray_in, -cam2lit_v, lit_v.normal, lit_v.v_is_mi(), lit_in_fspace)
                if cam_pdf > 0. and lit_pdf > 0.:   # if 
                    tr_con = self.track_ray(cam2lit_v, cam_v.pos, length)   # calculate transmittance from vertex to camera
                    if tr_con.max() > 0.:           # if not occluded
                        fr_cam = self.eval(cam_v.obj_id, cam_v.ray_in, cam2lit_v, cam_v.normal, cam_v.v_is_mi(), cam_in_fspace)
                        fr_lit = self.eval(lit_v.obj_id, lit_v.ray_in, -cam2lit_v, lit_v.normal, lit_v.v_is_mi(), lit_in_fspace)
                        # Geometry term: two cosine is in fr_xxx, length^{-2} is directly computed here
                        le = cam_v.beta * (fr_cam / cam_pdf) * (tr_con / (length * length)) * (fr_lit / lit_pdf) * lit_v.beta
        weight = 0.
        if le.max > 0.:             # zero-contribution will not have MIS weight
            weight = 1.0
            if sid + tid != 2:      # for path with only two vertices, forward and backward is the same
                weight = self.bdpt_mis_weight(sampled_v, vertex_sampled, sid, tid)
        return ti.select(weight > 0., le / weight, ZERO_V3), raster_p

    @ti.func
    def update_endpoint(self, cam_end: ti.template(), lit_end: ti.template(), i: int, j: int, idx_t: int, idx_s: int):
        # s + t > 2, since s + t == 2 will not enter `mis_weight`, and s + t < 2 will not have path connection
        if idx_s >= 0:                  # If lit_end is not null vertex
            lit_end.pdf(self, cam_end, self.null)  # Update camera endpoint
            if idx_t >= 1:
                cam_end.pdf(self, self.cam_paths[i, j, idx_t - 1], lit_end)
                cam_end.pdf(self, lit_end, self.null)       # 1
            if idx_s >= 1:
                lit_end.pdf(self, self.light_paths[i, j, idx_s - 1], cam_end)       # 2
        else:           # idx_t must >= 2       
            # if the camera hits an emitter
            cam_end.pdf_light_origin(self)
            if idx_t >= 1:
                cam_end.pdf_light(self, self.cam_paths[i, j, idx_t - 1])

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
            if idx_s < 0:
                self.update_endpoint(self.cam_paths[i, j, idx_t], self.null, i, j, idx_t, idx_s)
            else:           # Here idx_s can not be 0 (sid == 1 will go to s_sampled logic above)
                self.update_endpoint(self.cam_paths[i, j, idx_t], self.light_paths[i, j, idx_s], i, j, idx_t, idx_s)

        ri = ti.select(t_sampled, sampled_v.pdf_ratio(), self.cam_paths[i, j, idx_t].pdf_ratio())
        # Avoid indexing one vertex of cam_paths / light_paths twice 
        not_delta = False
        if idx_t > 0 and not self.cam_paths[i, j, idx_t - 1].is_delta():
            not_delta = True
            sum_ri += ri
        idx_t -= 1
        while idx_t > 0:
            ri *= self.cam_paths[i, j, idx_t].pdf_ratio()
            next_not_delta = not self.cam_paths[i, j, idx_t - 1].is_delta()
            if not_delta and next_not_delta:
                sum_ri += ri
            not_delta = next_not_delta
            idx_t -= 1

        if idx_s >= 0:                        # sid can be 0, 
            ri = ti.select(s_sampled, sampled_v.pdf_ratio(), self.light_paths[i, j, idx_s].pdf_ratio())
            not_delta = False
            if idx_s > 0 and not self.light_paths[i, j, idx_s - 1].is_delta():
                not_delta = True
                sum_ri += ri
            idx_s -= 1
            while idx_s > 0:
                ri *= self.light_paths[i, j, idx_s].pdf_ratio()
                next_not_delta = not self.light_paths[i, j, idx_s - 1].is_delta()
                if not_delta and next_not_delta:
                    sum_ri += ri
                not_delta = next_not_delta
                idx_s -= 1

        # Recover from the backup values
        for idx in ti.static(range(2)):
            if tid - 1 - idx >= 0: self.cam_paths[i, j, tid - 1 - idx].pdf_bwd = backup[idx]
        for idx in ti.static(range(2)):
            if sid - 1 - idx >= 0: self.light_paths[i, j, sid - 1 - idx].pdf_bwd = backup[idx + 2]

        return 1. / (1. + sum_ri)

    @staticmethod
    @ti.func
    def interact_mode(is_mi: int, hit_light: int):
        return ti.select(is_mi, VERTEX_MEDIUM, ti.select(hit_light >= 0, VERTEX_EMITTER, VERTEX_SURFACE))
    
    @ti.func
    def rasterize_pinhole(self, ray_d: vec3):
        """ For path with only one camera vertex, ray should be re-rasterized to the film
            ray_d is pointing into the camera, therefore should be negated
        """
        valid_raster = False
        raster_p = vec2i([-1, -1])
        local_ray = self.inv_cam_r @ (-ray_d)
        z = local_ray[2]
        if z > 0:
            local_ray /= z
            pi = int(self.half_w + 0.5 - local_ray[0] / self.inv_focal)
            pj = int(self.half_h + 0.5 + local_ray[1] / self.inv_focal)
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
        we = 0.0
        pdf = 0.0
        raster_p = vec2i([-1, -1])
        dot_normal = -tm.dot(ray_d, self.cam_normal)
        if dot_normal > 0.:
            raster_p, is_valid = self.rasterize_pinhole(ray_d)
            if is_valid:        # not valid --- outside of imaging plane
                # For pinhole camera, lens area is 1., this pdf is already in sa measure 
                pdf = depth * depth / dot_normal
                we = 1.0 / (self.A * dot_normal * dot_normal)
        return we, pdf, raster_p
    
    @ti.func
    def pdf_camera(self, ray_d: vec3):
        """ PDF camera does not rasterize point, we assume that the point is within view frustum
            Implementation see reference: PBR-Book chapter 16-1, equation 16.2
            https://www.pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/The_Path-Space_Measurement_Equation#eq:importance-sa
        """
        pdf_pos = 1.
        pdf_dir = 1.
        dot_normal = tm.dot(ray_d, self.cam_normal)
        if dot_normal > 0.:     # No need to rasterize again
            pdf_dir /=  (self.A * ti.pow(dot_normal, 3))
        else:
            pdf_pos = 0.
            pdf_dir = 0.
        return pdf_pos, pdf_dir
    
    @staticmethod
    @ti.func
    def convert_density(pdf: float, this_pos: vec3, next_nv: vec3, next_pos: vec3, next_mi:int):
        """ Convert solid angle density to unit area density
            next_nv, next_pos, next_mi: normal vector / position / is_mi for the next vertex
        """
        diff_vec = next_pos - this_pos
        inv_norm2 = 1. / diff_vec.norm_sqr()
        pdf *= inv_norm2
        if not next_mi:
            pdf *= ti.abs(tm.dot(next_nv, diff_vec * ti.sqrt(inv_norm2)))
        return pdf
    
    def summary(self):
        print(f"[INFO] BDPT Finished rendering. SPP = {self.cnt[None]}. Rendering time: {self.clock.toc():.3f} s")
    