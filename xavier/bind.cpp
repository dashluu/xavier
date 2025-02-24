#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "bind.h"

PYBIND11_MODULE(xavier, m)
{
    init_xv_module(m);
}

void init_xv_module(py::module_ &m)
{
    py::class_<Shape>(m, "Shape")
        .def(py::init<const std::vector<uint64_t> &>(), "view"_a)
        .def("offset", &Shape::get_offset)
        .def("view", &Shape::get_view)
        .def("stride", &Shape::get_stride)
        .def("contiguous_stride", &Shape::get_contiguous_stride)
        .def("elms_per_dim", &Shape::get_elms_per_dim)
        .def("ndim", &Shape::get_ndim)
        .def("numel", &Shape::get_numel)
        .def("__eq__", &Shape::operator==, "shape"_a)
        .def("__neq__", &Shape::operator!=, "shape"_a)
        .def("__getitem__", [](const Shape &shape, const py::object &obj)
             { return vslice<uint64_t>(shape.get_view(), obj); }, "dim"_a)
        .def("__str__", &Shape::str)
        .def("__len__", &Shape::get_ndim);

    py::class_<Dtype>(m, "Dtype")
        .def(py::init<const std::string &, uint64_t>(), "name"_a, "size"_a)
        .def("name", &Dtype::get_name)
        .def("size", &Dtype::get_size)
        .def("__eq__", &Dtype::operator==, "dtype"_a)
        .def("__neq__", &Dtype::operator!=, "dtype"_a)
        .def("__hash__", [](const Dtype &dtype)
             { return std::hash<Dtype>()(dtype); })
        .def("__str__", &Dtype::str);

    m.attr("f16") = &f16;
    m.attr("f32") = &f32;
    m.attr("f64") = &f64;
    m.attr("i8") = &i8;
    m.attr("i16") = &i16;
    m.attr("i32") = &i32;
    m.attr("i64") = &i64;
    m.attr("b8") = &b8;
    m.attr("binary_dtypes") = &binary_dtypes;
    m.attr("num_dtypes") = &num_dtypes;

    py::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("MPS", DeviceType::MPS);

    py::class_<Device>(m, "Device")
        .def("type", &Device::get_type)
        .def("idx", &Device::get_idx)
        .def("__eq__", &Device::operator==)
        .def("__neq__", &Device::operator!=);

    m.attr("device0") = &device0;

    py::class_<Array, std::shared_ptr<Array>>(m, "Array", py::buffer_protocol())
        .def(py::init<const Shape &, const Dtype &, const Device &>(), "shape"_a, "dtype"_a = f32, "device"_a = device0)
        .def_buffer([](Array &arr) -> py::buffer_info
                    { return array_to_buffer(arr); })
        .def("id", &Array::get_id)
        .def("shape", &Array::get_shape)
        .def("dtype", &Array::get_dtype)
        .def("device", &Array::get_device)
        .def("op", &Array::get_op)
        .def("grad", &Array::get_grad)
        .def("ptr_", &Array::get_ptr)
        .def("access_", &Array::access_, "Accesses the kth element in the array.", "k"_a)
        .def("is_contiguous", &Array::is_contiguous)
        .def("numel", &Array::get_numel)
        .def("ndim", &Array::get_ndim)
        .def("itemsize", &Array::get_itemsize)
        .def("nbytes", &Array::get_nbytes)
        .def("alloc", &Array::alloc)
        .def("__str__", &Array::str)
        .def("__len__", [](const Array &arr)
             { return arr.get_shape()[0]; })
        .def("reshape", &Array::reshape, "Reshapes the array to the given view in-place.", "view"_a)
        .def("__getitem__", [](Array &arr, const py::object &obj)
             { return arr.slice(get_arr_ranges(arr, obj)); }, "obj"_a)
        .def_static("arange", &Array::arange, "Creates a new array containing an algebraic sequence of integers.", "view"_a, "start"_a, "step"_a, "dtype"_a = f32, "device"_a = device0)
        .def_static("constant", &Array::constant, "Creates a new array containing the same constant value.", "view"_a, "c"_a, "dtype"_a = f32, "device"_a = device0)
        .def("__add__", &Array::add, "rhs"_a)
        .def("__sub__", &Array::sub, "rhs"_a)
        .def("__mul__", &Array::mul, "rhs"_a)
        .def("__truediv__", &Array::div, "rhs"_a)
        .def("exp", &Array::exp)
        .def_static("from_buffer", &array_from_buffer, "Creates a 1D array from buffer without copying.", "buff"_a, "device"_a = device0);

    py::class_<Graph, std::unique_ptr<Graph, py::nodelete>>(m, "Graph")
        .def("root", &Graph::get_root)
        .def("__str__", &Graph::str)
        .def("forward", &Graph::forward)
        .def("backward", &Graph::backward);

#ifdef __APPLE__
    py::class_<MTLGraph, Graph, std::unique_ptr<MTLGraph>>(m, "MTLGraph")
        .def(py::init<std::shared_ptr<Array>, std::shared_ptr<MTLContext>>(), "root"_a, "ctx"_a);
    py::class_<MTLContext, std::shared_ptr<MTLContext>>(m, "MTLContext")
        .def(py::init<const std::string &>(), "lib_path"_a);
#endif
}

bool is_buff_contiguous(py::buffer_info &buff_info)
{
    auto view = buff_info.shape;
    std::vector<py::ssize_t> contiguous_stride(view.size());
    uint64_t s = 1;
    for (int i = view.size() - 1; i >= 0; i--)
    {
        contiguous_stride[i] = s * buff_info.itemsize;
        s *= view[i];
    }
    return buff_info.strides == contiguous_stride;
}

std::shared_ptr<Array> array_from_buffer(py::buffer &buff, const Device &device)
{
    auto buff_info = buff.request();
    if (!is_buff_contiguous(buff_info))
    {
        throw std::invalid_argument("Buffer is not contiguous.");
    }
    uint64_t numel = buff_info.size;
    Shape shape(0, {numel}, {1});
    uint8_t *ptr = static_cast<uint8_t *>(buff_info.ptr);
    return Array::from_buff(ptr, shape, descriptors_to_dtypes.at(buff_info.format), device);
}

py::buffer_info array_to_buffer(Array &arr)
{
    if (!arr.is_contiguous())
    {
        throw std::invalid_argument("Array is not contiguous.");
    }
    return py::buffer_info(
        arr.get_ptr(),
        arr.get_itemsize(),
        dtypes_to_descriptors[arr.get_dtype()],
        1,
        {arr.get_numel()},
        {arr.get_itemsize()});
}

template <class T>
std::vector<T> vslice(const std::vector<T> &v, const py::object &obj)
{
    std::vector<T> result;
    auto len = v.size();
    // obj can be an int or a slice
    if (py::isinstance<py::int_>(obj))
    {
        auto idx = map_idx(len, obj.cast<int64_t>());
        result.push_back(v[idx]);
    }
    else
    {
        auto range = slice_to_range(len, obj);
        for (uint64_t i = range.start; i < range.stop; i += range.step)
        {
            result.push_back(v[i]);
        }
    }
    return result;
}

std::vector<Range> get_arr_ranges(const Array &arr, const py::object &obj)
{
    std::vector<Range> ranges;
    auto &shape = arr.get_shape();
    // obj can be an int, a slice, or a tuple of ints or slices
    if (py::isinstance<py::int_>(obj))
    {
        auto idx = map_idx(shape[0], obj.cast<int64_t>());
        ranges.emplace_back(idx, idx + 1, 1);
        for (int i = 1; i < shape.get_ndim(); i++)
        {
            ranges.emplace_back(0, shape[i], 1);
        }
    }
    else if (py::isinstance<py::slice>(obj))
    {
        ranges.push_back(slice_to_range(shape[0], obj));
        for (int i = 1; i < shape.get_ndim(); i++)
        {
            ranges.emplace_back(0, shape[i], 1);
        }
    }
    else
    {
        // obj is a tuple of ints or slices
        auto tuple = obj.cast<py::tuple>();
        if (tuple.size() > shape.get_ndim())
        {
            throw std::invalid_argument("The number of ranges exceeds the number of dimensions: " +
                                        std::to_string(tuple.size()) + " and " + std::to_string(shape.get_ndim()) + ".");
        }
        for (int i = 0; i < tuple.size(); i++)
        {
            auto elm = tuple[i];
            // elm is a tuple of ints or slices
            if (py::isinstance<py::int_>(elm))
            {
                auto idx = map_idx(shape[i], elm.cast<int64_t>());
                ranges.emplace_back(idx, idx + 1, 1);
            }
            else
            {
                ranges.push_back(slice_to_range(shape[i], elm));
            }
        }
        for (int i = tuple.size(); i < shape.get_ndim(); i++)
        {
            ranges.emplace_back(0, shape[i], 1);
        }
    }
    return ranges;
}

uint64_t map_idx(int64_t len, int64_t idx)
{
    if (idx < -len || idx >= len)
    {
        throw std::out_of_range("Index out of range: " + std::to_string(idx) + " not in [-" + std::to_string(len) + ", " + std::to_string(len) + ")");
    }
    return idx < 0 ? idx + len : idx;
}

Range slice_to_range(int64_t len, const py::object &obj)
{
    auto slice = obj.cast<py::slice>();
    auto start = slice.attr("start").is_none() ? 0 : map_idx(len, slice.attr("start").cast<int64_t>());
    auto stop = slice.attr("stop").is_none() ? len : map_idx(len, slice.attr("stop").cast<int64_t>());
    auto step = slice.attr("step").is_none() ? 1 : slice.attr("step").cast<int64_t>();
    return Range(start, stop, step);
}
