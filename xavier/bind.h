#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "core/iter.h"
#include "core/graph.h"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace xv::core;

void init_xv_module(py::module_ &);
bool is_buff_contiguous(py::buffer_info &buff_info);
std::shared_ptr<Array> array_from_buffer(py::buffer &buff, const Device &device);
py::buffer_info array_to_buffer(Array &arr);
template <class T>
std::vector<T> vslice(const std::vector<T> &v, const py::object &obj);
std::vector<Range> get_arr_ranges(const Array &arr, const py::object &obj);
uint64_t map_idx(int64_t len, int64_t idx);
Range slice_to_range(int64_t len, const py::object &obj);

inline auto f32_fmt = py::format_descriptor<float>::format();
inline auto f64_fmt = py::format_descriptor<double>::format();
inline auto i16_fmt = py::format_descriptor<int16_t>::format();
inline auto i32_fmt = py::format_descriptor<int32_t>::format();
inline auto i64_fmt = py::format_descriptor<int64_t>::format();
inline auto i8_fmt = py::format_descriptor<int8_t>::format();

inline std::unordered_map<Dtype, std::string> dtypes_to_descriptors = {
    {f32, f32_fmt},
    {f64, f64_fmt},
    {i8, i8_fmt},
    {i16, i16_fmt},
    {i32, i32_fmt},
    {i64, i64_fmt},
    {b8, i8_fmt}};

inline std::unordered_map<std::string, Dtype> descriptors_to_dtypes = []
{
    std::unordered_map<std::string, Dtype> m;
    for (const auto &pair : dtypes_to_descriptors)
    {
        if (pair.second != i8_fmt)
        {
            m.emplace(pair.second, pair.first);
            m.emplace("<" + pair.second, pair.first);
            m.emplace(">" + pair.second, pair.first);
        }
    }
    m.emplace(i8_fmt, i8);
    m.emplace("<" + i8_fmt, i8);
    m.emplace(">" + i8_fmt, i8);
    // Boolean in numpy
    m.emplace("?", b8);
    m.emplace("<?", b8);
    m.emplace(">?", b8);
    // For the buffer format, do not remove this
    m.emplace("B", i8);
    return m;
}();
