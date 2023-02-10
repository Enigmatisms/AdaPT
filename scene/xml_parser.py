"""
    Parsing scene XML file (simple one)
    currently, only BSDF, path tracer settings, emitter settings and .obj file
    configurations are supported
    @author Qianyue He (Enigmatisms)
    @date 2023.1.18
"""
import os
import sys
sys.path.append("..")

import xml.etree.ElementTree as xet

from typing import List
from numpy import ndarray as Arr

from bxdf.brdf import BRDF_np
from bxdf.bsdf import BSDF_np

from scene.obj_loader import *
from scene.world import World_np
from scene.obj_desc import ObjDescriptor
from scene.general_parser import get, transform_parse, parse_sphere_element

# import emitters
from emitters.point import PointSource
from emitters.rect_area import RectAreaSource
from emitters.directional import DirectionalSource

from utils.tools import timing

__VERSION__   = "1.1"
__MAPPING__   = {"integer": int, "float": float, "string": str, "boolean": lambda x: True if x.lower() == "true" else False}

"""
    Actually I think in Taichi, we can leverage SSDS:
    AABB and triangles are on the same level (attached to the same node)
    level 1    AABB1 (tri1 tri2 tri3)     AABB2 (tri4 tri5 tri6)     AABB3 (tri4 tri5 tri6)
"""

def update_emitter_config(emitter_config: List, area_lut: dict):
    for i, emitter in enumerate(emitter_config):
        if i in area_lut:
            emitter.inv_area = 1. / area_lut[i]
            emitter.attached = True
        else:
            if emitter.type == "rect_area":
                emitter.inv_area = 1. / (emitter.l1 * emitter.l2)
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
        source = None
        if emitter_type == "point":
            source = PointSource(elem)
        elif emitter_type == "rect_area":
            source = RectAreaSource(elem)
        elif emitter_type == "directional":
            source = DirectionalSource(elem)
        if source is not None:
            if source.id in source_id_dict:
                raise ValueError(f"Two sources with same id {source.id} will result in conflicts")
            source_id_dict[source.id] = len(sources)
            sources.append(source)
    return sources, source_id_dict

def parse_wavefront(directory: str, obj_list: List[xet.Element], bsdf_dict: dict, emitter_dict: dict) -> List[Arr]:
    """
        Parsing wavefront obj file (filename) from list of xml nodes    
    """
    all_objs = []
    # Some emitters will be attached to objects, to sample the object-attached emitters
    # We need to calculate surface area of the object mesh first (asuming each triangle has similar area)
    attached_area_dict = {}
    for elem in obj_list:
        trans_r, trans_t = None, None                           # transform
        obj_type = 0
        if elem.get("type") == "obj":
            filepath_child      = elem.find("string")
            meshes, normals     = load_obj_file(os.path.join(directory, filepath_child.get("value")))
            transform_child     = elem.find("transform")
            if transform_child is not None:
                trans_r, trans_t    = transform_parse(transform_child)
                meshes, normals     = apply_transform(meshes, normals, trans_r, trans_t)
        else:                   # CURRENTLY, only sphere is supported
            meshes, normals = parse_sphere_element(elem)
            obj_type = 1
        ref_childs     = elem.findall("ref")        
        bsdf_item      = None
        emitter_ref_id = -1
        for ref_child in ref_childs:
            ref_type = ref_child.get("type")
            if ref_type == "material":
                bsdf_item = bsdf_dict[ref_child.get("id")]
            elif ref_type == "emitter":
                emitter_ref_id = emitter_dict[ref_child.get("id")]
                attached_area_dict[emitter_ref_id] = calculate_surface_area(meshes, obj_type)
        if bsdf_item is None:
            raise ValueError("Object should be attached with a BSDF for now since no default one implemented yet.")
        all_objs.append(ObjDescriptor(meshes, normals, bsdf_item, trans_r, trans_t, emit_id = emitter_ref_id, _type = obj_type))
    return all_objs, attached_area_dict

def parse_bxdf(bxdf_list: List[xet.Element]):
    """
        Parsing BSDF / BRDF from xml file
        return: dict
    """
    results = dict()
    for bxdf_node in bxdf_list:
        bxdf_id = bxdf_node.get("id")
        bxdf_type = bxdf_node.tag
        if bxdf_type == "brdf":
            bxdf = BRDF_np(bxdf_node)
        else:
            bxdf = BSDF_np(bxdf_node)
        if bxdf_id in results:
            print(f"Warning: BXDF {bxdf_id} re-defined in XML file. Overwriting the existing BXDF.")
        results[bxdf_id] = bxdf
    return results

def parse_world(world_elem: xet.Element):
    world = World_np(world_elem)
    if world_elem is None:
        print("Warning: world element not found in xml file. Using default world settings:")
        print(world)
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
    assert(len(film_elems) >= 2)        # at least width, height and sample count (meaningless for direct component tracer)
    sensor_config["film"]       = {}
    for elem in film_elems:
        if elem.tag in __MAPPING__:
            name = elem.get("name")
            sensor_config["film"][name] = get(elem, "value", __MAPPING__[elem.tag])
    return sensor_config

@timing()
def mitsuba_parsing(directory: str, file: str):
    xml_file = os.path.join(directory, file)
    print(f"Parsing XML file from '{xml_file}'")
    node_tree = xet.parse(xml_file)
    root_node = node_tree.getroot()
    version_tag = root_node.attrib["version"]
    if not version_tag == __VERSION__:
        raise ValueError(f"Unsupported version {version_tag}. Only '{__VERSION__}' is supported right now.")  
    # Export list of dict for emitters / dict for other secen settings and film settings / list for obj files
    bxdf_nodes       = root_node.findall("bsdf") + root_node.findall("brdf")
    emitter_nodes    = root_node.findall("emitter")
    shape_nodes      = root_node.findall("shape")
    sensor_node      = root_node.find("sensor")
    world_node       = root_node.find("world")
    assert(sensor_node)
    emitter_configs, \
    emitter_dict     = parse_emitters(emitter_nodes)
    bsdf_dict        = parse_bxdf(bxdf_nodes)
    meshes, area_lut = parse_wavefront(directory, shape_nodes, bsdf_dict, emitter_dict)
    configs          = parse_global_sensor(sensor_node)
    configs['world'] = parse_world(world_node)
    emitter_configs  = update_emitter_config(emitter_configs, area_lut)
    return emitter_configs, bsdf_dict, meshes, configs

if __name__ == "__main__":
    emitter_configs, bsdf_configs, meshes, configs = mitsuba_parsing("../inputs/", "cbox/complex.xml")
    print(emitter_configs, bsdf_configs, meshes, configs)