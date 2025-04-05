#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "array.h"

namespace xb = xv::bind;

PYBIND11_MODULE(xavier, m)
{
    init_xv_module(m);
}

void init_xv_module(py::module_ &m)
{
    py::class_<xc::Id>(m, "Id")
        .def("data", &xc::Id::get_data)
        .def("__str__", &xc::Id::str)
        .def("__repr__", &xc::Id::str);

    py::class_<xc::Shape>(m, "Shape")
        .def(py::init<const xc::ShapeView &>(), "view"_a)
        .def("offset", &xc::Shape::get_offset)
        .def("view", &xc::Shape::get_view)
        .def("stride", &xc::Shape::get_stride)
        .def("contiguous_stride", &xc::Shape::get_contiguous_stride)
        .def("elms_per_dim", &xc::Shape::get_elms_per_dim)
        .def("ndim", &xc::Shape::get_ndim)
        .def("numel", &xc::Shape::get_numel)
        .def("is_contiguous", &xc::Shape::is_contiguous)
        .def("broadcast_to", &xc::Shape::broadcast_to, "target"_a)
        .def("broadcast", &xc::Shape::broadcast, "rhs"_a)
        .def("broadcastable", &xc::Shape::broadcastable, "rhs"_a)
        .def("broadcastable_to", &xc::Shape::broadcastable_to, "target"_a)
        .def("matmul_broadcastable", &xc::Shape::matmul_broadcastable, "rhs"_a)
        .def("remove", &xc::Shape::remove, "dim"_a)
        .def("permute", &xc::Shape::permute, "order"_a)
        .def("__eq__", &xc::Shape::operator==, "shape"_a)
        .def("__neq__", &xc::Shape::operator!=, "shape"_a)
        .def("__getitem__", [](const xc::Shape &shape, const py::object &obj)
             { return xb::vslice<xc::usize>(shape.get_view(), obj); }, "dim"_a)
        .def("__str__", &xc::Shape::str)
        .def("__len__", &xc::Shape::get_ndim);

    py::class_<xc::Dtype>(m, "Dtype")
        .def(py::init<const std::string &, xc::usize>(), "name"_a, "size"_a)
        .def("name", &xc::Dtype::get_name)
        .def("size", &xc::Dtype::get_size)
        .def("__eq__", &xc::Dtype::operator==, "dtype"_a)
        .def("__neq__", &xc::Dtype::operator!=, "dtype"_a)
        .def("__hash__", [](const xc::Dtype &dtype)
             { return std::hash<xc::Dtype>()(dtype); })
        .def("__str__", &xc::Dtype::str);

    m.attr("f16") = &xc::f16;
    m.attr("f32") = &xc::f32;
    m.attr("i8") = &xc::i8;
    m.attr("i16") = &xc::i16;
    m.attr("i32") = &xc::i32;
    m.attr("b8") = &xc::b8;

    py::enum_<xc::DeviceType>(m, "DeviceType")
        .value("CPU", xc::DeviceType::CPU)
        .value("MPS", xc::DeviceType::MPS);

    py::class_<xc::Device>(m, "Device")
        .def("type", &xc::Device::get_type)
        .def("idx", &xc::Device::get_idx)
        .def("__eq__", &xc::Device::operator==)
        .def("__neq__", &xc::Device::operator!=);

    m.attr("device0") = &xc::device0;

    py::class_<xc::Array, xc::ArrayPtr>(m, "Array", py::buffer_protocol())
        .def(py::init<const xc::Shape &, const xc::Dtype &, const xc::Device &>(), "shape"_a, "dtype"_a = xc::f32, "device"_a = xc::device0)
        .def_buffer([](xc::Array &arr) -> py::buffer_info
                    { return xb::array_to_buffer(arr); })
        .def("id", &xc::Array::get_id, "Returns the id of the array.")
        .def("shape", &xc::Array::get_shape, "Returns the shape of the array.")
        .def("offset", &xc::Array::get_offset, "Returns the offset of the array.")
        .def("view", &xc::Array::get_view, "Returns the view of the array.")
        .def("stride", &xc::Array::get_stride, "Returns the stride of the array.")
        .def("dtype", &xc::Array::get_dtype, "Returns the data type of the array.")
        .def("device", &xc::Array::get_device, "Returns the device that the array is allocated on.")
        .def_readonly("grad", &xc::Array::grad, "Accesses the gradient of the array.")
        .def("ptr", [](const xc::Array &arr)
             { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(arr.get_ptr())); }, "Returns a pointer to the data of the array.")
        .def("strided_idx", &xc::Array::strided_idx, "Accesses the kth element in the array.", "k"_a)
        .def("is_contiguous", &xc::Array::is_contiguous, "Checks if the array is contiguous.")
        .def("numel", &xc::Array::get_numel, "Returns the number of elements in the array.")
        .def("ndim", &xc::Array::get_ndim, "Returns the number of dimensions in the array.")
        .def("itemsize", &xc::Array::get_itemsize, "Returns the size of each element in the array in bytes.")
        .def("nbytes", &xc::Array::get_nbytes, "Returns the total size of the array in bytes.")
        .def("__str__", &xc::Array::str)
        .def("__repr__", &xc::Array::str)
        .def("__len__", [](const xc::Array &arr)
             { return arr.get_shape()[0]; })
        .def("reshape", &xc::Array::reshape, "Reshapes the array to the given view in-place.", "view"_a)
        .def("broadcast", &xc::Array::broadcast, "Broadcasts the array based on the given view.", "view"_a)
        .def("broadcast_to", &xc::Array::broadcast_to, "Broadcasts the array to the given view.", "view"_a)
        .def("as_contiguous", &xc::Array::as_contiguous, "Creates a new contiguous array with the same as elements as the current array if the current array is not contiguous, otherwise, returns the current array.")
        .def("__getitem__", [](xc::Array &arr, const py::object &obj)
             { return arr.slice(xb::get_arr_ranges(arr, obj)); }, "obj"_a)
        .def_static("arange", &xc::Array::arange, "Creates a new array containing an algebraic sequence of integers.", "view"_a, "start"_a, "step"_a, "dtype"_a = xc::f32, "device"_a = xc::device0, "constant"_a = false)
        .def_static("full", &xb::full, "Creates a new array containing the same value.", "view"_a, "c"_a, "dtype"_a = xc::f32, "device"_a = xc::device0, "constant"_a = false)
        .def_static("zeros", &xc::Array::zeros, "Creates a new array containing zeros.", "view"_a, "dtype"_a = xc::f32, "device"_a = xc::device0, "constant"_a = false)
        .def_static("ones", &xc::Array::ones, "Creates a new array containing ones.", "view"_a, "dtype"_a = xc::f32, "device"_a = xc::device0, "constant"_a = false)
        .def_static("full_like", &xb::full_like, "Creates a new array containing the same value that has the same shape and data type as the given array.", "arr"_a, "c"_a, "device"_a = xc::device0, "constant"_a = false)
        .def_static("zeros_like", &xc::Array::zeros_like, "Creates a new array similar to the given array containing zeros.", "arr"_a, "device"_a = xc::device0, "constant"_a = false)
        .def_static("ones_like", &xc::Array::ones_like, "Creates a new array similar to the given array containing ones.", "arr"_a, "device"_a = xc::device0, "constant"_a = false)
        .def("__add__", &xb::add, "rhs"_a)
        .def("__sub__", &xb::sub, "rhs"_a)
        .def("__mul__", &xb::mul, "rhs"_a)
        .def("__rmul__", &xb::mul, "rhs"_a)
        .def("__truediv__", &xb::div, "rhs"_a)
        .def("__iadd__", &xb::self_add, "rhs"_a)
        .def("__isub__", &xb::self_sub, "rhs"_a)
        .def("__imul__", &xb::self_mul, "rhs"_a)
        .def("__itruediv__", &xb::self_div, "rhs"_a)
        .def("__matmul__", &xb::matmul, "rhs"_a)
        .def("__eq__", &xb::eq, "rhs"_a)
        .def("__ne__", &xb::neq, "rhs"_a)
        .def("__gt__", &xb::gt, "rhs"_a)
        .def("__ge__", &xb::geq, "rhs"_a)
        .def("__lt__", &xb::lt, "rhs"_a)
        .def("__le__", &xb::leq, "rhs"_a)
        .def("exp", &xc::Array::exp, "Element-wise exponential function.", "in_place"_a = false)
        .def("log", &xc::Array::log, "Element-wise natural logarithm function.", "in_place"_a = false)
        .def("__neg__", &xc::Array::neg)
        .def("neg", &xc::Array::neg, "Element-wise negation.", "in_place"_a = false)
        .def("identity", &xc::Array::identity, "Copies data of the current array to a new array.")
        .def("recip", &xc::Array::recip, "Element-wise reciprocal.", "in_place"_a = false)
        .def("sq", &xc::Array::sq, "Element-wise square.", "in_place"_a = false)
        .def("sqrt", &xc::Array::sqrt, "Element-wise square root.", "in_place"_a = false)
        .def("permute", &xb::permute, "Permutes the dimensions of the array according to the given order.", "order"_a)
        .def("T", &xb::T, "Transposes the array.", "start_dim"_a = 0, "end_dim"_a = -1)
        .def("flatten", &xb::flatten, "Flattens the array.", "start_dim"_a = 0, "end_dim"_a = -1)
        .def("sum", &xb::sum, "Computes the sum of the array elements in given dimensions.", "dims"_a = std::vector<py::int_>())
        .def("max", &xb::max, "Computes the maximum of the array elements in given dimensions.", "dims"_a = std::vector<py::int_>())
        .def_static("from_buffer", &xb::array_from_buffer, "Creates a 1D array from buffer without copying.", "buff"_a, "device"_a = xc::device0, "constant"_a = false)
        .def_static("from_numpy", &xb::array_from_numpy, "Creates an array from numpy array without copying.", "np_arr"_a, "device"_a = xc::device0, "constant"_a = false)
        .def("numpy", &xb::array_to_numpy, "Converts the array to a numpy array.");

    py::class_<xg::Graph, std::unique_ptr<xg::Graph, py::nodelete>>(m, "Graph")
        .def("root", &xg::Graph::get_root)
        .def("__str__", &xg::Graph::str)
        .def("compile", &xg::Graph::compile)
        .def("forward", &xg::Graph::forward)
        .def("backward", &xg::Graph::backward);

#ifdef __APPLE__
    py::class_<xg::MTLGraph, xg::Graph, std::unique_ptr<xg::MTLGraph>>(m, "MTLGraph")
        .def(py::init<xc::ArrayPtr, std::shared_ptr<xm::MTLContext>>(), "root"_a, "ctx"_a);
    py::class_<xm::MTLContext, std::shared_ptr<xm::MTLContext>>(m, "MTLContext")
        .def(py::init<const std::string &>(), "lib_path"_a);
#endif

    m.def("add", &xb::m_add, "Element-wise addition.", "lhs"_a, "rhs"_a);
    m.def("sub", &xb::m_sub, "Element-wise subtraction.", "lhs"_a, "rhs"_a);
    m.def("mul", &xb::m_mul, "Element-wise multiplication.", "lhs"_a, "rhs"_a);
    m.def("div", &xb::m_div, "Element-wise division.", "lhs"_a, "rhs"_a);
    m.def("self_add", &xb::m_self_add, "In-place element-wise addition.", "lhs"_a, "rhs"_a);
    m.def("self_sub", &xb::m_self_sub, "In-place element-wise subtraction.", "lhs"_a, "rhs"_a);
    m.def("self_mul", &xb::m_self_mul, "In-place element-wise multiplication.", "lhs"_a, "rhs"_a);
    m.def("self_div", &xb::m_self_div, "In-place element-wise division.", "lhs"_a, "rhs"_a);
    m.def("matmul", &xb::m_matmul, "Matrix multiplication.", "lhs"_a, "rhs"_a);
    m.def("eq", &xb::m_eq, "Element-wise equality.", "lhs"_a, "rhs"_a);
    m.def("neq", &xb::m_neq, "Element-wise inequality.", "lhs"_a, "rhs"_a);
    m.def("lt", &xb::m_lt, "Element-wise less than.", "lhs"_a, "rhs"_a);
    m.def("gt", &xb::m_gt, "Element-wise greater than.", "lhs"_a, "rhs"_a);
    m.def("leq", &xb::m_leq, "Element-wise less than or equal to.", "lhs"_a, "rhs"_a);
    m.def("geq", &xb::m_geq, "Element-wise greater than or equal to.", "lhs"_a, "rhs"_a);
    m.def("exp", &xb::m_exp, "Element-wise exponential function.", "arr"_a, "in_place"_a = false);
    m.def("log", &xb::m_log, "Element-wise natural logarithm function.", "arr"_a, "in_place"_a = false);
    m.def("neg", &xb::m_neg, "Element-wise negation.", "arr"_a, "in_place"_a = false);
    m.def("identity", &xb::m_identity, "Copies data of the current array to a new array.");
    m.def("recip", &xb::m_recip, "Element-wise reciprocal.", "arr"_a, "in_place"_a = false);
    m.def("sq", &xb::m_sq, "Element-wise square.", "arr"_a, "in_place"_a = false);
    m.def("sqrt", &xb::m_sqrt, "Element-wise square root.", "arr"_a, "in_place"_a = false);
    m.def("permute", &xb::m_permute, "Permutes the dimensions of the array according to the given order.", "arr"_a, "order"_a);
    m.def("T", &xb::m_T, "Transposes the array.", "arr"_a, "start_dim"_a = 0, "end_dim"_a = -1);
    m.def("flatten", &xb::m_flatten, "Flattens the array.", "arr"_a, "start_dim"_a = 0, "end_dim"_a = -1);
    m.def("sum", &xb::m_sum, "Computes the sum of the array elements in given dimensions.", "arr"_a, "dims"_a = std::vector<py::int_>());
    m.def("max", &xb::m_max, "Computes the maximum of the array elements in given dimensions.", "arr"_a, "dims"_a = std::vector<py::int_>());
}