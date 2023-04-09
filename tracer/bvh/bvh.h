#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <iostream>

/** 
 * The first objective is to correctly implement the passing of parameter
 * from python to C++ and vice versa, the first specific task is to
 * load wavefront obj in python and pass it to C++ code for printing
 * and vice versa, C++ should return some processed object to python 
*/

namespace py = pybind11;

/**
 * Here comes a problem: When we do ray-mesh intersection, we want to know: 
 * (1) which primitives to intersect (2) object information (3) normal information
 * If the primitives are reordered, then we should store obj_id in here and sort them along with
 * primitives, to keep the mapping of primitives and objects
 */
std::tuple<py::list, py::list> bvh_build(
    const py::array_t<float>& obj_array, const py::array_t<int>& obj_info,
    const py::array_t<float>& world_min, const py::array_t<int>& world_max
);

PYBIND11_MODULE(bvh_cpp, m) {

    m.doc() = "Build SAH-BVH tree via cpp backend\n"
    "Input: obj_array of shape (N_faces, 3, 3), N_faces corresponds to number of primitives (triangles, spheres)\n"
    "Input: obj_info of shape (2, N_obj), N_obj corresponds to number of object\n"
    "Input: world_min: world AABB min vertex\n"
    "Input: world_min: world AABB max vertex"
    ;

    m.def("bvh_build", &bvh_build, "Build SAH-BVH tree via cpp backend.");
}

PYBIND11_MODULE(bvh_cpp, m) {

    m.doc() = "Build SAH-BVH tree via cpp backend\n"
    "Input: obj_array of shape (N_faces, 3, 3), N_faces corresponds to number of primitives (triangles, spheres)\n"
    "Input: obj_info of shape (2, N_obj), N_obj corresponds to number of object\n"
    "Input: world_min: world AABB min vertex\n"
    "Input: world_min: world AABB max vertex"
    ;

    m.def("bvh_build", &bvh_build, "Build SAH-BVH tree via cpp backend.");

    py::class_<LinearBVH>(m, "LinearBVH")
        .def_readonly("mini", &LinearBVH::mini)
        .def_readonly("maxi", &LinearBVH::maxi)
        .def_readonly("obj_idx", &LinearBVH::obj_idx)
        .def_readonly("prim_idx", &LinearBVH::prim_idx)
        .def(py::init<int>());

    py::class_<LinearBVH>(m, "LinearNode")
        .def_readonly("mini", &LinearNode::mini)
        .def_readonly("maxi", &LinearNode::maxi)
        .def_readonly("base", &LinearNode::base)
        .def_readonly("prim_cnt", &LinearNode::prim_cnt)
        .def_readonly("rc_offset", &LinearNode::rc_offset)
        .def_readonly("all_offset", &LinearNode::all_offset)
        .def(py::init<int>());
}