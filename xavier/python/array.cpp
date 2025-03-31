#include "array.h"

namespace xv::bind
{
	ArrayPtr full(const std::vector<uint64_t> &view, const py::object &c, const Dtype &dtype, const Device &device, bool constant)
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

	ArrayPtr full_like(ArrayPtr arr, const py::object &c, const Device &device, bool constant)
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

	ArrayPtr unary(const py::object &operand_obj, const std::function<ArrayPtr(ArrayPtr, bool)> &f, bool in_place)
	{
		return f(obj_to_arr(operand_obj, device0), in_place);
	}

	ArrayPtr m_binary(const py::object &lhs_obj, const py::object &rhs_obj, const std::function<ArrayPtr(ArrayPtr, ArrayPtr)> &f)
	{
		ArrayPtr lhs_arr;
		ArrayPtr rhs_arr;
		if (is_scalar(lhs_obj))
		{
			rhs_arr = obj_to_arr(rhs_obj, device0);
			lhs_arr = obj_to_arr(lhs_obj, rhs_arr->get_device());
		}
		else
		{
			lhs_arr = obj_to_arr(lhs_obj, device0);
			rhs_arr = obj_to_arr(rhs_obj, lhs_arr->get_device());
		}
		return f(lhs_arr, rhs_arr);
	}

	ArrayPtr binary(ArrayPtr lhs_arr, const py::object &rhs_obj, const std::function<ArrayPtr(ArrayPtr, ArrayPtr)> &f)
	{
		return f(lhs_arr, obj_to_arr(rhs_obj, lhs_arr->get_device()));
	}

	ArrayPtr reduce(const py::object &operand_obj, const std::function<ArrayPtr(ArrayPtr)> &f)
	{
		return f(obj_to_arr(operand_obj, device0));
	}
}