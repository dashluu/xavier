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
        .def("is_contiguous", &Shape::is_contiguous)
        .def("broadcast_to", &Shape::broadcast_to, "target"_a)
        .def("broadcast", &Shape::broadcast, "rhs"_a)
        .def("broadcastable", &Shape::broadcastable, "rhs"_a)
        .def("broadcastable_to", &Shape::broadcastable_to, "target"_a)
        .def("matmul_broadcastable", &Shape::matmul_broadcastable, "rhs"_a)
        .def("matmul_broadcast", &Shape::matmul_broadcast, "rhs"_a, "output_result"_a = true)
        .def("remove", &Shape::remove, "dim"_a)
        .def("permute", &Shape::permute, "order"_a)
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
    m.attr("i8") = &i8;
    m.attr("i16") = &i16;
    m.attr("i32") = &i32;
    m.attr("b8") = &b8;

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
        .def("id", &Array::get_id, "Returns the id of the array.")
        .def("shape", &Array::get_shape, "Returns the shape of the array.")
        .def("dtype", &Array::get_dtype, "Returns the data type of the array.")
        .def("device", &Array::get_device, "Returns the device that the array is allocated on.")
        .def_readonly("cum_grad", &Array::cum_grad, "Accesses the cumulated gradient of the array.")
        .def_readonly("grad", &Array::grad, "Accesses the gradient of the array.")
        .def("ptr", [](const Array &arr)
             { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(arr.get_ptr())); }, "Returns a pointer to the data of the array")
        .def("access_", &Array::access_, "Accesses the kth element in the array.", "k"_a)
        .def("is_contiguous", &Array::is_contiguous, "Checks if the array is contiguous.")
        .def("numel", &Array::get_numel, "Returns the number of elements in the array.")
        .def("ndim", &Array::get_ndim, "Returns the number of dimensions in the array.")
        .def("itemsize", &Array::get_itemsize, "Returns the size of each element in the array in bytes.")
        .def("nbytes", &Array::get_nbytes, "Returns the total size of the array in bytes.")
        .def("__str__", &Array::str)
        .def("__repr__", &Array::str)
        .def("__len__", [](const Array &arr)
             { return arr.get_shape()[0]; })
        .def("reshape", &Array::reshape, "Reshapes the array to the given view in-place.", "view"_a)
        .def("copy", &Array::copy, "Copies data of the current array to a new array.")
        .def("broadcast", &Array::broadcast, "Broadcasts the array based on the given view", "view"_a)
        .def("broadcast_to", &Array::broadcast_to, "Broadcasts the array to the given view", "view"_a)
        .def("as_contiguous", &Array::as_contiguous, "Creates a new contiguous array with the same as elements as the current array if the current array is not contiguous, otherwise, returns the current array.")
        .def("__getitem__", [](Array &arr, const py::object &obj)
             { return arr.slice(get_arr_ranges(arr, obj)); }, "obj"_a)
        .def_static("arange", &Array::arange, "Creates a new array containing an algebraic sequence of integers.", "view"_a, "start"_a, "step"_a, "dtype"_a = f32, "device"_a = device0, "constant"_a = false)
        .def_static("full", [](const std::vector<uint64_t> &view, const py::object &c, const Dtype &dtype, const Device &device, bool constant)
                    { return full(view, c, dtype, device, constant); }, "Creates a new array containing the same value.", "view"_a, "c"_a, "dtype"_a = f32, "device"_a = device0, "constant"_a = false)
        .def_static("zeros", &Array::zeros, "Creates a new array containing zeros.", "view"_a, "dtype"_a = f32, "device"_a = device0, "constant"_a = false)
        .def_static("ones", &Array::ones, "Creates a new array containing ones.", "view"_a, "dtype"_a = f32, "device"_a = device0, "constant"_a = false)
        .def_static("full_like", [](std::shared_ptr<Array> arr, const py::object &c, const Device &device, bool constant)
                    { return full_like(arr, c, device, constant); }, "Creates a new array containing the same value that has the same shape and data type as the given array.", "arr"_a, "c"_a, "device"_a = device0, "constant"_a = false)
        .def_static("zeros_like", &Array::zeros_like, "Creates a new array similar to the given array containing zeros.", "arr"_a, "device"_a = device0, "constant"_a = false)
        .def_static("ones_like", &Array::ones_like, "Creates a new array similar to the given array containing ones.", "arr"_a, "device"_a = device0, "constant"_a = false)
        .def("__add__", &Array::add, "rhs"_a)
        .def("__sub__", &Array::sub, "rhs"_a)
        .def("__mul__", [](std::shared_ptr<Array> lhs, const py::object &rhs)
             { return lhs->mul(obj_to_arr(rhs, lhs->get_device())); }, "rhs"_a)
        .def("__rmul__", [](std::shared_ptr<Array> lhs, const py::object &rhs)
             { return lhs->mul(obj_to_arr(rhs, lhs->get_device())); }, "rhs"_a)
        .def("__truediv__", &Array::div, "rhs"_a)
        .def("__iadd__", &Array::self_add, "rhs"_a)
        .def("__isub__", &Array::self_sub, "rhs"_a)
        .def("__imul__", &Array::self_mul, "rhs"_a)
        .def("__itruediv__", &Array::self_div, "rhs"_a)
        .def("__matmul__", &Array::matmul, "rhs"_a)
        .def("__eq__", [](std::shared_ptr<Array> lhs, const py::object &rhs)
             { return lhs->eq(obj_to_arr(rhs, lhs->get_device())); }, "rhs"_a)
        .def("__ne__", [](std::shared_ptr<Array> lhs, const py::object &rhs)
             { return lhs->neq(obj_to_arr(rhs, lhs->get_device())); }, "rhs"_a)
        .def("__gt__", [](std::shared_ptr<Array> lhs, const py::object &rhs)
             { return lhs->gt(obj_to_arr(rhs, lhs->get_device())); }, "rhs"_a)
        .def("__ge__", [](std::shared_ptr<Array> lhs, const py::object &rhs)
             { return lhs->geq(obj_to_arr(rhs, lhs->get_device())); }, "rhs"_a)
        .def("__lt__", [](std::shared_ptr<Array> lhs, const py::object &rhs)
             { return lhs->lt(obj_to_arr(rhs, lhs->get_device())); }, "rhs"_a)
        .def("__le__", [](std::shared_ptr<Array> lhs, const py::object &rhs)
             { return lhs->leq(obj_to_arr(rhs, lhs->get_device())); }, "rhs"_a)
        .def("exp", &Array::exp, "Element-wise exponential function", "in_place"_a = false)
        .def("log", &Array::log, "Element-wise natural logarithm function", "in_place"_a = false)
        .def("__neg__", [](std::shared_ptr<Array> arr)
             { return arr->neg(); })
        .def("neg", &Array::neg, "Element-wise negation", "in_place"_a = false)
        .def("recip", &Array::recip, "Element-wise reciprocal", "in_place"_a = false)
        .def("sq", &Array::sq, "Element-wise square", "in_place"_a = false)
        .def("sqrt", &Array::sqrt, "Element-wise square root", "in_place"_a = false)
        .def_static("from_buffer", &array_from_buffer, "Creates a 1D array from buffer without copying.", "buff"_a, "device"_a = device0, "constant"_a = false);

    py::class_<Graph, std::unique_ptr<Graph, py::nodelete>>(m, "Graph")
        .def("root", &Graph::get_root)
        .def("__str__", &Graph::str)
        .def("compile", &Graph::compile)
        .def("forward", &Graph::forward)
        .def("backward", &Graph::backward);

#ifdef __APPLE__
    py::class_<MTLGraph, Graph, std::unique_ptr<MTLGraph>>(m, "MTLGraph")
        .def(py::init<std::shared_ptr<Array>, std::shared_ptr<MTLContext>>(), "root"_a, "ctx"_a);
    py::class_<MTLContext, std::shared_ptr<MTLContext>>(m, "MTLContext")
        .def(py::init<const std::string &>(), "lib_path"_a);
#endif

    m.def("add", [](const py::object &lhs, const py::object &rhs)
          { return binary(lhs, rhs, [](std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs)
                          { return lhs->add(rhs); }); }, "Element-wise addition", "lhs"_a, "rhs"_a);

    m.def("sub", [](const py::object &lhs, const py::object &rhs)
          { return binary(lhs, rhs, [](std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs)
                          { return lhs->sub(rhs); }); }, "Element-wise subtraction", "lhs"_a, "rhs"_a);

    m.def("mul", [](const py::object &lhs, const py::object &rhs)
          { return binary(lhs, rhs, [](std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs)
                          { return lhs->mul(rhs); }); }, "Element-wise multiplication", "lhs"_a, "rhs"_a);

    m.def("div", [](const py::object &lhs, const py::object &rhs)
          { return binary(lhs, rhs, [](std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs)
                          { return lhs->div(rhs); }); }, "Element-wise division", "lhs"_a, "rhs"_a);

    m.def("matmul", [](const py::object &lhs, const py::object &rhs)
          { return binary(lhs, rhs, [](std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs)
                          { return lhs->matmul(rhs); }); }, "Matrix multiplication", "lhs"_a, "rhs"_a);

    m.def("eq", [](const py::object &lhs, const py::object &rhs)
          { return binary(lhs, rhs, [](std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs)
                          { return lhs->eq(rhs); }); }, "Element-wise equality", "lhs"_a, "rhs"_a);

    m.def("neq", [](const py::object &lhs, const py::object &rhs)
          { return binary(lhs, rhs, [](std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs)
                          { return lhs->neq(rhs); }); }, "Element-wise inequality", "lhs"_a, "rhs"_a);

    m.def("lt", [](const py::object &lhs, const py::object &rhs)
          { return binary(lhs, rhs, [](std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs)
                          { return lhs->lt(rhs); }); }, "Element-wise less than", "lhs"_a, "rhs"_a);

    m.def("gt", [](const py::object &lhs, const py::object &rhs)
          { return binary(lhs, rhs, [](std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs)
                          { return lhs->gt(rhs); }); }, "Element-wise greater than", "lhs"_a, "rhs"_a);

    m.def("leq", [](const py::object &lhs, const py::object &rhs)
          { return binary(lhs, rhs, [](std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs)
                          { return lhs->leq(rhs); }); }, "Element-wise less than or equal to", "lhs"_a, "rhs"_a);

    m.def("geq", [](const py::object &lhs, const py::object &rhs)
          { return binary(lhs, rhs, [](std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs)
                          { return lhs->geq(rhs); }); }, "Element-wise greater than or equal to", "lhs"_a, "rhs"_a);

    m.def("exp", [](const py::object &arr, bool in_place)
          { return unary(arr, [](std::shared_ptr<Array> arr, bool in_place)
                         { return arr->exp(in_place); }, in_place); }, "Element-wise exponential function", "arr"_a, "in_place"_a = false);

    m.def("log", [](const py::object &arr, bool in_place)
          { return unary(arr, [](std::shared_ptr<Array> arr, bool in_place)
                         { return arr->log(in_place); }, in_place); }, "Element-wise natural logarithm function", "arr"_a, "in_place"_a = false);

    m.def("neg", [](const py::object &arr, bool in_place)
          { return unary(arr, [](std::shared_ptr<Array> arr, bool in_place)
                         { return arr->neg(in_place); }, in_place); }, "Element-wise negation", "arr"_a, "in_place"_a = false);

    m.def("recip", [](const py::object &arr, bool in_place)
          { return unary(arr, [](std::shared_ptr<Array> arr, bool in_place)
                         { return arr->recip(in_place); }, in_place); }, "Element-wise reciprocal", "arr"_a, "in_place"_a = false);

    m.def("sq", [](const py::object &arr, bool in_place)
          { return unary(arr, [](std::shared_ptr<Array> arr, bool in_place)
                         { return arr->sq(in_place); }, in_place); }, "Element-wise square", "arr"_a, "in_place"_a = false);

    m.def("sqrt", [](const py::object &arr, bool in_place)
          { return unary(arr, [](std::shared_ptr<Array> arr, bool in_place)
                         { return arr->sqrt(in_place); }, in_place); }, "Element-wise square root", "arr"_a, "in_place"_a = false);
}

std::shared_ptr<Array> full(const std::vector<uint64_t> &view, const py::object &c, const Dtype &dtype, const Device &device, bool constant)
{
    if (py::isinstance<py::float_>(c))
    {
        return Array::full(view, c.cast<float>(), dtype, device, constant);
    }
    else if (py::isinstance<py::int_>(c) || py::isinstance<py::bool_>(c))
    {
        return Array::full(view, c.cast<int>(), dtype, device, constant);
    }
    throw PybindInvalidArgumentType(get_pyclass(c), "float, int, bool");
}

std::shared_ptr<Array> full_like(std::shared_ptr<Array> arr, const py::object &c, const Device &device, bool constant)
{
    if (py::isinstance<py::float_>(c))
    {
        return Array::full_like(arr, c.cast<float>(), device, constant);
    }
    else if (py::isinstance<py::int_>(c) || py::isinstance<py::bool_>(c))
    {
        return Array::full_like(arr, c.cast<int>(), device, constant);
    }
    throw PybindInvalidArgumentType(get_pyclass(c), "float, int, bool");
}

std::shared_ptr<Array> unary(const py::object &obj, const std::function<std::shared_ptr<Array>(std::shared_ptr<Array>, bool)> &f, bool in_place)
{
    return f(obj_to_arr(obj, device0), in_place);
}

std::shared_ptr<Array> binary(const py::object &obj1, const py::object &obj2, const std::function<std::shared_ptr<Array>(std::shared_ptr<Array>, std::shared_ptr<Array>)> &f)
{
    std::shared_ptr<Array> arr1;
    std::shared_ptr<Array> arr2;
    if (is_scalar(obj1))
    {
        arr2 = obj_to_arr(obj2, device0);
        arr1 = obj_to_arr(obj1, arr2->get_device());
    }
    else
    {
        arr1 = obj_to_arr(obj1, device0);
        arr2 = obj_to_arr(obj2, arr1->get_device());
    }
    return f(arr1, arr2);
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

std::shared_ptr<Array> array_from_buffer(py::buffer &buff, const Device &device, bool constant)
{
    auto buff_info = buff.request();
    if (!is_buff_contiguous(buff_info))
    {
        throw std::invalid_argument("Buffer is not contiguous.");
    }
    uint64_t numel = buff_info.size;
    uint64_t nbytes = buff_info.size * buff_info.itemsize;
    Shape shape(0, {numel}, {1});
    uint8_t *ptr = static_cast<uint8_t *>(buff_info.ptr);
    return Array::from_buff(ptr, nbytes, shape, descriptors_to_dtypes.at(buff_info.format), device, constant);
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
    // obj must be an int or a slice
    if (py::isinstance<py::int_>(obj))
    {
        auto idx = map_idx(len, obj.cast<int64_t>());
        result.push_back(v[idx]);
        return result;
    }
    else if (py::isinstance<py::slice>(obj))
    {
        auto range = slice_to_range(len, obj);
        for (uint64_t i = range.start; i < range.stop; i += range.step)
        {
            result.push_back(v[i]);
        }
        return result;
    }
    throw PybindInvalidArgumentType(get_pyclass(obj), "int, slice");
}

std::vector<Range> get_arr_ranges(const Array &arr, const py::object &obj)
{
    std::vector<Range> ranges;
    auto &shape = arr.get_shape();
    // obj can be an int, a slice, or a sequence of ints or slices
    if (py::isinstance<py::int_>(obj))
    {
        auto idx = map_idx(shape[0], obj.cast<int64_t>());
        ranges.emplace_back(idx, idx + 1, 1);
        for (int i = 1; i < shape.get_ndim(); i++)
        {
            ranges.emplace_back(0, shape[i], 1);
        }
        return ranges;
    }
    else if (py::isinstance<py::slice>(obj))
    {
        ranges.push_back(slice_to_range(shape[0], obj));
        for (int i = 1; i < shape.get_ndim(); i++)
        {
            ranges.emplace_back(0, shape[i], 1);
        }
        return ranges;
    }
    else if (py::isinstance<py::sequence>(obj) && !py::isinstance<py::str>(obj))
    {
        // Object is a sequence but not a string
        auto sequence = obj.cast<py::sequence>();
        if (sequence.size() > shape.get_ndim())
        {
            throw std::invalid_argument("The number of ranges exceeds the number of dimensions: " + std::to_string(sequence.size()) + " and " + std::to_string(shape.get_ndim()) + ".");
        }
        for (int i = 0; i < sequence.size(); i++)
        {
            auto elm = sequence[i];
            // elm must be a sequence of ints or slices
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
        for (int i = sequence.size(); i < shape.get_ndim(); i++)
        {
            ranges.emplace_back(0, shape[i], 1);
        }
        return ranges;
    }
    throw PybindInvalidArgumentType(get_pyclass(obj), "int, slice, list, tuple");
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
    if (!py::isinstance<py::slice>(obj))
    {
        throw PybindInvalidArgumentType(get_pyclass(obj), "slice");
    }
    auto slice = obj.cast<py::slice>();
    auto start = slice.attr("start").is_none() ? 0 : map_idx(len, slice.attr("start").cast<int64_t>());
    auto stop = slice.attr("stop").is_none() ? len : map_idx(len, slice.attr("stop").cast<int64_t>());
    auto step = slice.attr("step").is_none() ? 1 : slice.attr("step").cast<int64_t>();
    return Range(start, stop, step);
}

std::shared_ptr<Array> obj_to_arr(const py::object &obj, const Device &device_if_scalar)
{
    if (py::isinstance<Array>(obj))
    {
        return obj.cast<std::shared_ptr<Array>>();
    }
    else if (py::isinstance<py::float_>(obj))
    {
        return Array::full(obj.cast<float>(), f32, device_if_scalar);
    }
    else if (py::isinstance<py::int_>(obj))
    {
        return Array::full(obj.cast<int>(), i32, device_if_scalar);
    }
    else if (py::isinstance<py::bool_>(obj))
    {
        return Array::full(obj.cast<int>(), b8, device_if_scalar);
    }
    throw PybindInvalidArgumentType(get_pyclass(obj), "float, int, bool, Array");
}

bool is_scalar(const py::object &obj)
{
    return py::isinstance<py::float_>(obj) || py::isinstance<py::int_>(obj) || py::isinstance<py::bool_>(obj);
}
