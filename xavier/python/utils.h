#pragma once

#include "bind.h"

namespace xv::bind
{
	bool is_buff_contiguous(py::buffer_info &buff_info);

	xc::ArrayPtr array_from_buffer(py::buffer &buff, const xc::Device &device, bool constant);

	py::buffer_info array_to_buffer(xc::Array &arr);

	xc::ArrayPtr array_from_numpy(py::array &np_arr, const xc::Device &device, bool constant);

	py::array array_to_numpy(xc::Array &arr);

	std::vector<xc::Range> get_arr_ranges(const xc::Array &arr, const py::object &obj);

	uint64_t map_idx(int64_t len, int64_t idx);

	xc::Range slice_to_range(int64_t len, const py::object &obj);

	inline std::string get_pyclass(const py::object &obj) { return obj.attr("__class__").cast<py::str>().cast<std::string>(); }

	// This does not mean array will be constructed on the specified device
	// It means that if the object is a scalar, the array corresponding to that scalar will be constructed on the specified device
	xc::ArrayPtr obj_to_arr(const py::object &obj, const xc::Device &device_if_scalar);

	bool is_scalar(const py::object &obj);

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
		throw xc::PybindInvalidArgumentType(get_pyclass(obj), "int, slice");
	}
}