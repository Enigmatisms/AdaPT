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
 * 
 * TODO: decide what to output here, the output parameter?
 */
void bvh_build(const py::array_t<float>& obj_array, const py::array_t<int>& id_array);

void wavefront_input(const py::array_t<float>& obj_array, std::vector<Eigen::Matrix3f>& meshes);

PYBIND11_MODULE(bvh_cpp, m) {

    m.doc() = "Build SAH-BVH tree via cpp backend\nInput: obj_array of shape (N_faces, 3, 3), N_faces corresponds to ";

    m.def("bvh_build", &bvh_build, "Build SAH-BVH tree via cpp backend.");
}