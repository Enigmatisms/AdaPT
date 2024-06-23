/**
 * @file vdb2numpy.cpp
 * @author Qianyue He
 * @brief Load mitsuba3 .vol volume data to numpy
 * @version 0.1
 * @date 2024-06-21
 * @copyright Copyright (c) 2024
 */
#include <omp.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Volume data intermediate representation
struct VolumeData {
    int xres, yres, zres;
    int channels;
    std::vector<float> data;

    inline size_t size(bool force_mono_color = false) const noexcept {
        size_t ch = force_mono_color ? 1 : channels;
        return ch * xres * yres * zres;
    }

    inline auto shape() const noexcept {
        return std::tuple<int, int, int, int>(xres, yres, zres, channels);
    }
};

bool readVolumeData(const std::string& filename, VolumeData& volume) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    char header[4];
    file.read(header, 4);
    if (header[0] != 'V' || header[1] != 'O' || header[2] != 'L' || header[3] != 3) {
        std::cerr << "Invalid file format" << std::endl;
        return false;
    }

    int encoding;
    file.read(reinterpret_cast<char*>(&encoding), sizeof(int));
    if (encoding != 1) {
        std::cerr << "Unsupported encoding" << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(&volume.xres), sizeof(int));
    file.read(reinterpret_cast<char*>(&volume.yres), sizeof(int));
    file.read(reinterpret_cast<char*>(&volume.zres), sizeof(int));

    file.read(reinterpret_cast<char*>(&volume.channels), sizeof(int));
    if (volume.channels != 1 && volume.channels != 3 && volume.channels != 6) {
        std::cerr << "Unsupported number of channels" << std::endl;
        return false;
    }

    file.seekg(24, std::ios::cur); // Skip the bounding box

    int numVoxels = volume.xres * volume.yres * volume.zres * volume.channels;
    volume.data.resize(numVoxels);
    file.read(reinterpret_cast<char*>(volume.data.data()), numVoxels * sizeof(float));

    file.close();
    return true;
}

// mitsuba3 vol to numpy
auto loadVol2Numpy(const std::string& filename, bool force_mono_color = false) {
    VolumeData volume;
    readVolumeData(filename, volume);

    py::array_t<float> vol_numpy(volume.size(force_mono_color));
    float* const data_ptr = vol_numpy.mutable_data(0);
    if (volume.channels == 1) {
        #pragma omp parallel for num_threads(4)
        for (int z = 0; z < volume.zres; ++z) {
            int zy_base = z * volume.yres;
            for (int y = 0; y < volume.yres; ++y) {
                int zyx_base = (zy_base + y) * volume.xres;
                for (int x = 0; x < volume.xres; ++x) {
                    data_ptr[zyx_base + x] = volume.data[zyx_base + x];
                }
            }
        }
    } else if (volume.channels == 3) {
        #pragma omp parallel for num_threads(4)
        for (int z = 0; z < volume.zres; ++z) {
            int zy_base = z * volume.yres;
            for (int y = 0; y < volume.yres; ++y) {
                int zyx_base = (zy_base + y) * volume.xres;
                if (!force_mono_color) {
                    for (int x = 0; x < volume.xres; ++x) {
                        int index = (zyx_base + x) * 3;
                        data_ptr[index] = volume.data[index];
                        data_ptr[index + 1] = volume.data[index + 1];
                        data_ptr[index + 2] = volume.data[index + 2];
                    }
                } else {
                    for (int x = 0; x < volume.xres; ++x) {
                        int index = zyx_base + x;
                        data_ptr[index] = volume.data[index * 3 + 1];
                    }
                }
            }
        }
    } else {
        std::cerr << "Grid channel: <" << volume.channels << "> is not supported, supported channels: [1, 3]" << std::endl;
    }

    if (force_mono_color)
        volume.channels = 1;
    
    return std::tuple(vol_numpy, volume.shape());
}

PYBIND11_MODULE(vol_loader, m) {
    m.doc() = "Volume grid (.vol / .vdb) loader (to numpy)\n";

    m.def("vol_file_to_numpy", &loadVol2Numpy, "Load volume grid from mitsuba3 .vol file (return numpy)\n"
        "Input: filename, [string] input path of the file\n"
        "Input: force_mono_color, [bool] whether to extract only one channel from the volume data, default = False (True only for testing)\n"
    );
}