#include "array.h"

namespace xv::bind
{
	xc::ArrayPtr full(const xc::ShapeView &view, const py::object &c, const xc::Dtype &dtype, const xc::Device &device, bool constant)
	{
		if (py::isinstance<py::float_>(c))
		{
			return xc::Array::full(view, c.cast<float>(), dtype, device, constant);
		}
		else if (py::isinstance<py::int_>(c) || py::isinstance<py::bool_>(c))
		{
			return xc::Array::full(view, c.cast<int>(), dtype, device, constant);
		}
		throw xc::PybindInvalidArgumentType(get_pyclass(c), "float, int, bool");
	}

	xc::ArrayPtr full_like(xc::ArrayPtr arr, const py::object &c, const xc::Device &device, bool constant)
	{
		if (py::isinstance<py::float_>(c))
		{
			return xc::Array::full_like(arr, c.cast<float>(), device, constant);
		}
		else if (py::isinstance<py::int_>(c) || py::isinstance<py::bool_>(c))
		{
			return xc::Array::full_like(arr, c.cast<int>(), device, constant);
		}
		throw xc::PybindInvalidArgumentType(get_pyclass(c), "float, int, bool");
	}

	xc::ArrayPtr unary(const py::object &py_operand, const std::function<xc::ArrayPtr(xc::ArrayPtr, bool)> &f, bool in_place)
	{
		return f(obj_to_arr(py_operand, xc::device0), in_place);
	}

	xc::ArrayPtr m_binary(const py::object &py_lhs, const py::object &py_rhs, const std::function<xc::ArrayPtr(xc::ArrayPtr, xc::ArrayPtr)> &f)
	{
		xc::ArrayPtr lhs_arr;
		xc::ArrayPtr rhs_arr;
		if (is_scalar(py_lhs))
		{
			rhs_arr = obj_to_arr(py_rhs, xc::device0);
			lhs_arr = obj_to_arr(py_lhs, rhs_arr->get_device());
		}
		else
		{
			lhs_arr = obj_to_arr(py_lhs, xc::device0);
			rhs_arr = obj_to_arr(py_rhs, lhs_arr->get_device());
		}
		return f(lhs_arr, rhs_arr);
	}

	xc::ArrayPtr binary(xc::ArrayPtr lhs_arr, const py::object &py_rhs, const std::function<xc::ArrayPtr(xc::ArrayPtr, xc::ArrayPtr)> &f)
	{
		return f(lhs_arr, obj_to_arr(py_rhs, lhs_arr->get_device()));
	}

	xc::ArrayPtr m_reduce(const py::object &py_operand,
						  const std::vector<py::int_> &py_dims,
						  const std::function<xc::ArrayPtr(xc::ArrayPtr, const std::vector<xc::usize> &)> &f)
	{
		return reduce(obj_to_arr(py_operand, xc::device0), py_dims, f);
	}

	xc::ArrayPtr reduce(xc::ArrayPtr operand,
						const std::vector<py::int_> &py_dims,
						const std::function<xc::ArrayPtr(xc::ArrayPtr, const std::vector<xc::usize> &)> &f)
	{
		std::vector<xc::usize> dims;
		for (auto &dim : py_dims)
		{
			dims.push_back(dim.cast<xc::usize>());
		}
		return f(operand, dims);
	}
}