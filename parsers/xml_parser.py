"""
    Parsing scene XML file (simple one)
    currently, only BSDF, path tracer settings, emitter settings and .obj file
    configurations are supported
    @author Qianyue He (Enigmatisms)
    @date 2023.1.18
"""
import os
import sys
import numpy as np
sys.path.append("..")

import xml.etree.ElementTree as xet

from typing import List
from numpy import ndarray as Arr

from bxdf.brdf import BRDF_np
from bxdf.bsdf import BSDF_np
from bxdf.texture import Texture_np
from bxdf.mixture import BxDFMixture_np

from parsers.obj_loader import *
from parsers.world import World_np
from parsers.obj_desc import ObjDescriptor
from parsers.texture_packing import image_packer
from parsers.general_parser import get, transform_parse, parse_sphere_element

# import emitters
from emitters.point import PointSource
from emitters.area import AreaSource
from emitters.spot import SpotSource
from emitters.collimated import CollimatedSource

from utils.tools import timing

from rich.console import Console
CONSOLE = Console(width = 128)

__VERSION__   = "1.1"
__MAPPING__   = {"integer": int, "float": float, "string": str, "boolean": lambda x: True if x.lower() == "true" else False}
__SOURCE_MAP__ = {"point": PointSource, "area": AreaSource, "spot": SpotSource, "collimated": CollimatedSource}

"""
    Actually I think in Taichi, we can leverage SSDS:
    AABB and triangles are on the same level (attached to the same node)
    level 1    AABB1 (tri1 tri2 tri3)     AABB2 (tri4 tri5 tri6)     AABB3 (tri4 tri5 tri6)
"""

def none_checker(value, prim_num, last_dim = 3):
    """ Check whether the value is None. If True, replace the value with a zero array """
    if value is None:
        return np.zeros((prim_num, 3, last_dim), dtype = np.float32)
    return value

def update_emitter_config(emitter_config: List, area_lut: dict):
    for i, emitter in enumerate(emitter_config):
        if i in area_lut:
            emitter.inv_area = 1. / area_lut[i]
            emitter.attached = True
        else:
            if emitter.type == "area":
                raise ValueError("Setting L1 / L2 for area light is deprecated a long ago. Please attach area light to an object.")
    return emitter_config

def parse_emitters(em_elem: list):
    """
        Parsing scene emitters from list of xml nodes \\
        only [Point], [Area], [Directional] are supported
    """
    sources = []
    source_id_dict = dict()
    for elem in em_elem:
        emitter_type = elem.get("type")
        source_type = __SOURCE_MAP__.get(emitter_type, None)
        source = None
        if source_type is None:
            raise ValueError(f"Source type '{emitter_type}' is not supported. Please check your XML settings.")
        source = source_type(elem)
        if source is not None:
            if source.id in source_id_dict:
                raise ValueError(f"Two sources with same id {source.id} will result in conflicts")
            source_id_dict[source.id] = len(sources)
            sources.append(source)
        else:
            raise ValueError(f"Source [{emitter_type}] is not implemented.")
    return sources, source_id_dict

def parse_mixture(mixture_nodes: List[xet.Element], bxdf_dict: dict, is_brdf: dict) -> List[BxDFMixture_np]:
    """ Parsing BxDF mixture and potentially process non-mixture BxDF 
        - bxdf_dict: [key: str, the id for bxdf in xml file], [value: int, index for the bxdf in the list]
        - is_brdf: dict, [key: str, the id for bxdf in xml file], [value: bool, the bxdf being BSDF/BRDF]

        we will know if we require a mixture, when the object is parsed. If a single BSDF is required both in 
        a mixture / non-mixture BSDF item, then we will 
        
    """
    mixture_dict = {}
    ref_cnt_dict = {key:0 for key in bxdf_dict.keys()}
    for mixture_node in mixture_nodes:
        mixture = BxDFMixture_np(mixture_node, bxdf_dict)
        for ref_id in mixture.ref_ids:
            ref_cnt_dict[ref_id] += 1
        if mixture.id in mixture_dict:
            CONSOLE.log(f"[yellow]Warning: [/yellow]Mixture <{mixture.id}> is defined multiple times. The old versions might be overwritten.")
        mixture_dict[mixture.id] = mixture      
    for bxdf_name in bxdf_dict.keys():
        mixture = BxDFMixture_np.from_single(bxdf_name, bxdf_dict[bxdf_name], is_brdf[bxdf_name])
        if mixture.id in mixture_dict:
            CONSOLE.log(f"[yellow]Warning: [/yellow]Looks like you have a single BSDF which has the same name as a mixture (id = <{mixture.id}>).")
            CONSOLE.log(f"Therefore mixture <{mixture.id}> is defined multiple times. The old versions might be overwritten.")
        mixture_dict[mixture.id] = mixture
    return mixture_dict

def parse_wavefront(
    directory: str, obj_list: List[xet.Element], mixture_dict: dict, 
    emitter_dict: dict, texture_dict: List[Texture_np]) -> List[Arr]:
    """ Parsing wavefront obj file (filename) from list of xml nodes """
    all_objs  = []
    all_prims = []
    all_uvs   = []
    all_normals = []
    all_v_norms = []
    indices = []            # indicating whether the current primitive is sphere or not 
    # Some emitters will be attached to objects, to sample the object-attached emitters
    # We need to calculate surface area of the object mesh first (asuming each triangle has similar area)
    attached_area_dict = {}
    has_vertex_normal = False
    cum_prim_num = 0
    for elem in obj_list:
        # vns: vertex normals / uvs: uv coordinates
        vns, uvs, trans_r, trans_t = None, None, None, None                           # uv_coordinates and transform
        obj_type = TRIANGLE_MESH
        if elem.get("type") == "obj":
            filepath_child       = elem.find("string")
            # get mesh / geometrical normal / shading normal / uv coords for texture
            meshes, normals, vns, uvs = extract_obj_info(os.path.join(directory, filepath_child.get("value")))
            transform_child      = elem.find("transform")
            if transform_child is not None:
                trans_r, trans_t    = transform_parse(transform_child)
                meshes, normals     = apply_transform(meshes, normals, trans_r, trans_t)
            if vns is not None:
                has_vertex_normal = True
        else:                   # CURRENTLY, only sphere is supported
            meshes, normals = parse_sphere_element(elem)
            obj_type = SPHERE
        ref_childs    = elem.findall("ref")        
        mixture_item     = None
        texture_group = {"albedo": None, "normal": None, "bump": None, "roughness": None}
        emit_ref_id   = -1

        # Currently, texture (groups) and object form bi-jection, since this way is simpler to implement
        # and can avoid id look up (memory ops might be very slow) 
        for ref_child in ref_childs:
            ref_type = ref_child.get("type")
            ref_id = ref_child.get("id")
            if ref_type == "mixture" or ref_type == "material":
                # single BSDF will be converted to a mixture with the same name
                mixture_item = mixture_dict[ref_id]
            elif ref_type == "emitter":
                emit_ref_id = emitter_dict[ref_id]
                attached_area_dict[emit_ref_id] = calculate_surface_area(meshes, obj_type)
            elif ref_type == "texture":
                ref_tag = ref_child.get("tag", None)
                if ref_tag == None:
                    ref_tag = "albedo"
                    CONSOLE.log(f"[yellow]Warning: BXDF[/yellow] Texture ref_id {ref_id} has no tag. Set default as 'albedo'.")
                elif ref_tag not in texture_group:
                    ref_tag = "albedo"
                    CONSOLE.log(f"[yellow]Warning: BXDF[/yellow] Texture ref_tag {ref_tag} not supported. Set default as 'albedo'.")
                # make sure texture group has corresponding tag
                if texture_dict[ref_tag] is None or ref_id not in texture_dict[ref_tag]:
                    raise KeyError(f"Texture id '{ref_id}' does not have tag '{ref_tag}' mapping, check if it is from other groups.")
                texture_group[ref_tag] = texture_dict[ref_tag][ref_id]
            
        if mixture_item is None:
            raise ValueError("Object should be attached with a BSDF for now since no default one implemented yet.")
        prim_num = meshes.shape[0] 
        if obj_type == SPHERE:      # padding to (1, 3, 3)
            meshes = np.concatenate((meshes, np.zeros((1, 1, 3), dtype=np.float32)), axis = -2)
            indices.append(cum_prim_num)
        all_prims.append(meshes)
        all_normals.append(normals)
        all_v_norms.append(none_checker(vns, prim_num))
        all_uvs.append(none_checker(uvs, prim_num, last_dim = 2))

        all_objs.append(ObjDescriptor(meshes, normals, mixture_item, vns, uvs, texture_group, trans_r, trans_t, emit_ref_id, obj_type))
        cum_prim_num += prim_num
    
    if indices:
        indices = np.int64(indices)
    else:
        indices = None
    all_uvs = np.concatenate(all_uvs, axis = 0).astype(np.float32)
    all_prims = np.concatenate(all_prims, axis = 0).astype(np.float32)
    all_normals = np.concatenate(all_normals, axis = 0).astype(np.float32)
    all_v_norms = np.concatenate(all_v_norms, axis = 0).astype(np.float32)
    array_info = {"primitives": all_prims, "indices": indices, "n_g": all_normals, "n_s": all_v_norms, "uvs": all_uvs}
    return array_info, all_objs, attached_area_dict, has_vertex_normal

def parse_bxdf(bxdf_list: List[xet.Element]):
    """
        Parsing BSDF / BRDF from xml file
        return: dict
    """
    bxdf_dict = dict()
    is_brdf = dict()
    for bxdf_node in bxdf_list:
        bxdf_id = bxdf_node.get("id")
        bxdf_type = bxdf_node.tag
        brdf_flag = True
        if bxdf_type == "brdf":
            bxdf = BRDF_np(bxdf_node)
        else:
            brdf_flag = False
            bxdf = BSDF_np(bxdf_node)
        if bxdf_id in bxdf_dict:
            CONSOLE.log(f"[yellow]Warning: BXDF[/yellow] {bxdf_id} [bold yellow]re-defined[/bold yellow] in XML file. Overwriting the existing BXDF.")
        bxdf_dict[bxdf_id] = bxdf
        is_brdf[bxdf_id] = brdf_flag
    index_dict = dict()
    for i, bxdf_key in enumerate(bxdf_dict.keys()):
        index_dict[bxdf_key] = i
    bxdf_list = list(bxdf_dict.values())
    return index_dict, bxdf_list, is_brdf

def parse_texture(texture_list: List[xet.Element]):
    """ Parsing Textures
        return Dict of Texture_np, containing four different types of mapping
    """
    if len(texture_list) == 0:
        return None, None
    textures = {"albedo": [], "normal": [], "bump": [], "roughness": []}
    for texture in texture_list:
        map_type = texture.get("tag", "albedo")
        textures[map_type].append(Texture_np(texture))
    # Do texture packing
    packed_textures = {}
    packed_imgs = {}
    for key, value in textures.items():
        if len(value) == 0:
            tex_img, tex_info = None, None
        else:
            tex_img, tex_info = image_packer(value)
        packed_imgs[key] = tex_img
        packed_textures[key] = tex_info
    return packed_imgs, packed_textures

def parse_world(world_elem: xet.Element):
    world = World_np(world_elem)
    if world_elem is None:
        CONSOLE.log("[yellow]Warning: world element not found in xml file. Using default world settings:")
        CONSOLE.log(world)
    return world

def parse_global_sensor(sensor_elem: xet.Element):
    """
        Parsing sensor (there can only be one sensor)
        Other global configs related to film, etc. are loaded here
    """
    sensor_config = {}
    for elem in sensor_elem:
        if elem.tag in __MAPPING__:
            name = elem.get("name")
            sensor_config[name] = get(elem, "value", __MAPPING__[elem.tag])

    sensor_config["transform"]  = transform_parse(sensor_elem.find("transform"))
    film_elems                  = sensor_elem.find("film").findall("integer")
    assert(len(film_elems) >= 2)        # at least width, height (meaningless for direct component tracer)
    sensor_config["film"]       = {}
    for elem in film_elems:
        if elem.tag in __MAPPING__:
            name = elem.get("name")
            sensor_config["film"][name] = get(elem, "value", __MAPPING__[elem.tag])
    return sensor_config

@timing()
def scene_parsing(directory: str, file: str):
    xml_file = os.path.join(directory, file)
    CONSOLE.log(f":fax: Parsing XML file from '{xml_file}'")
    node_tree = xet.parse(xml_file)
    root_node = node_tree.getroot()
    version_tag = root_node.attrib["version"]
    if not version_tag == __VERSION__:
        raise ValueError(f"Unsupported version {version_tag}. Only '{__VERSION__}' is supported right now.")  
    # Export list of dict for emitters / dict for other secen settings and film settings / list for obj files
    bxdf_nodes       = root_node.findall("bsdf") + root_node.findall("brdf")
    texture_nodes    = root_node.findall("texture")
    emitter_nodes    = root_node.findall("emitter")
    shape_nodes      = root_node.findall("shape")
    mixture_nodes    = root_node.findall("mixture")
    sensor_node      = root_node.find("sensor")
    world_node       = root_node.find("world")
    assert(sensor_node)
    emitter_configs, \
    emitter_dict     = parse_emitters(emitter_nodes)
    # FIXME: return BSDFs
    bxdf_dict, bxdfs, is_brdf = parse_bxdf(bxdf_nodes)
    # There might be single BSDF (without mixture reference)
    mixture_dict      = parse_mixture(mixture_nodes, bxdf_dict, is_brdf)
    teximgs, textures = parse_texture(texture_nodes)
    """ Texture mapping should be updateded (done):
    - [x] <ref type = "texture".../>. Now for each object, there can only be one ref for each type of reference
        But TODO: texture needs more that one (albedo map, normal map, bump map, roughness map), for other mappings
        I do not want to implement them. Therefore, 1-to-many mapping should be correctly established
    - [x] Loading from python to Taichi (for different kinds of mapping)
        Each mapping might needs a different packing, therefore we need different image packaging and texture info block
        For normal mapping, non-bidirectional renderers will be simple but not for BDPT
        roughness is of lower priority
    - [x] Speed up python->taichi conversion
    - [ ] mixture based BSDF parsing
    """
    array_info, all_objs, area_lut, has_vertex_normal \
                     = parse_wavefront(directory, shape_nodes, mixture_dict, emitter_dict, textures)
    configs          = parse_global_sensor(sensor_node)
    configs['world'] = parse_world(world_node)
    configs['packed_textures']   = teximgs
    configs['has_vertex_normal'] = has_vertex_normal
    emitter_configs  = update_emitter_config(emitter_configs, area_lut)
    return emitter_configs, array_info, all_objs, bxdfs, configs

if __name__ == "__main__":
    emitter_configs, array_info, all_objs, bxdfs, configs = scene_parsing("../scenes/", "cbox/complex.xml")
    print(emitter_configs, array_info.shape, all_objs, bxdfs, configs)