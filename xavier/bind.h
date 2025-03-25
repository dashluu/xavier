#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "core/iter.h"
#ifdef __APPLE__
#include "graph/mtl_graph.h"
#endif

namespace py = pybind11;
using namespace pybind11::literals;
using namespace xv::core;
using namespace xv::graph;

void init_xv_module(py::module_ &);
std::shared_ptr<Array> full(const std::vector<uint64_t> &view, const py::object &c, const Dtype &dtype, const Device &device, bool constant);
std::shared_ptr<Array> full_like(std::shared_ptr<Array> arr, const py::object &c, const Device &device, bool constant);
std::shared_ptr<Array> unary(const py::object &obj, const std::function<std::shared_ptr<Array>(std::shared_ptr<Array>, bool)> &f, bool in_place);
std::shared_ptr<Array> binary(const py::object &obj1, const py::object &obj2, const std::function<std::shared_ptr<Array>(std::shared_ptr<Array>, std::shared_ptr<Array>)> &f);
std::shared_ptr<Array> T(std::shared_ptr<Array> arr, int64_t start_dim, int64_t end_dim);
std::shared_ptr<Array> flatten(std::shared_ptr<Array> arr, int64_t start_dim, int64_t end_dim);
bool is_buff_contiguous(py::buffer_info &buff_info);
std::shared_ptr<Array> array_from_buffer(py::buffer &buff, const Device &device, bool constant);
py::buffer_info array_to_buffer(Array &arr);
std::shared_ptr<Array> array_from_numpy(py::array &np_arr, const Device &device, bool constant);
py::array array_to_numpy(Array &arr);
template <class T>
std::vector<T> vslice(const std::vector<T> &v, const py::object &obj);
std::vector<Range> get_arr_ranges(const Array &arr, const py::object &obj);
uint64_t map_idx(int64_t len, int64_t idx);
Range slice_to_range(int64_t len, const py::object &obj);
std::string get_pyclass(const py::object &obj) { return obj.attr("__class__").cast<py::str>().cast<std::string>(); }
// This does not mean array will be constructed on this device
// It means that if the object is a scalar, the array corresponding to that scalar will be constructed on this device
std::shared_ptr<Array> obj_to_arr(const py::object &obj, const Device &device_if_scalar);
bool is_scalar(const py::object &obj);

inline auto f32_fmt = py::format_descriptor<float>::format();
inline auto i16_fmt = py::format_descriptor<int16_t>::format();
inline auto i32_fmt = py::format_descriptor<int32_t>::format();
inline auto i64_fmt = py::format_descriptor<int64_t>::format();
inline auto i8_fmt = py::format_descriptor<int8_t>::format();

inline std::unordered_map<Dtype, std::string> dtypes_to_descriptors = {
    {f32, f32_fmt},
    {i8, i8_fmt},
    {i16, i16_fmt},
    {i32, i32_fmt},
    {b8, i8_fmt}};

inline std::unordered_map<std::string, Dtype> descriptors_to_dtypes = []
{
    std::unordered_map<std::string, Dtype> m;
    for (const auto &pair : dtypes_to_descriptors)
    {
        if (pair.second != i8_fmt)
        {
            const std::string sized_fmt = pair.second + std::to_string(pair.first.get_size());
            // Buffer format
            m.emplace("<" + pair.second, pair.first);
            m.emplace(">" + pair.second, pair.first);
            // Numpy array format
            m.emplace("<" + sized_fmt, pair.first);
            m.emplace(">" + sized_fmt, pair.first);
        }
    }
    // Byte in numpy
    m.emplace("<" + i8_fmt, i8);
    m.emplace(">" + i8_fmt, i8);
    m.emplace("|" + i8_fmt, i8);
    // Boolean in numpy
    m.emplace("<?", b8);
    m.emplace(">?", b8);
    m.emplace("|?", b8);
    // For the buffer format, do not remove this
    m.emplace("B", i8);
    return m;
}();
