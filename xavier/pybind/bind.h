#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../core/iter.h"
#ifdef __APPLE__
#include "../graph/mtl_graph.h"
#endif

namespace py = pybind11;
namespace xc = xv::core;
namespace xg = xv::graph;
namespace xm = xv::backend::metal;
using namespace pybind11::literals;

void init_xv_module(py::module_ &);

namespace xv::bind
{
    inline auto f32_fmt = py::format_descriptor<float>::format();
    inline auto i16_fmt = py::format_descriptor<int16_t>::format();
    inline auto i32_fmt = py::format_descriptor<int32_t>::format();
    inline auto i64_fmt = py::format_descriptor<int64_t>::format();
    inline auto i8_fmt = py::format_descriptor<int8_t>::format();

    inline std::unordered_map<xc::Dtype, std::string> dtypes_to_descriptors = {
        {xc::f32, f32_fmt},
        {xc::i8, i8_fmt},
        {xc::i16, i16_fmt},
        {xc::i32, i32_fmt},
        {xc::b8, i8_fmt}};

    inline std::unordered_map<std::string, xc::Dtype> descriptors_to_dtypes = []
    {
        std::unordered_map<std::string, xc::Dtype> m;
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
        m.emplace("<" + i8_fmt, xc::i8);
        m.emplace(">" + i8_fmt, xc::i8);
        m.emplace("|" + i8_fmt, xc::i8);
        // Boolean in numpy
        m.emplace("<?", xc::b8);
        m.emplace(">?", xc::b8);
        m.emplace("|?", xc::b8);
        // For the buffer format, do not remove this
        m.emplace("B", xc::i8);
        return m;
    }();
}
