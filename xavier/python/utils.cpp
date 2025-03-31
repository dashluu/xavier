#include "utils.h"

namespace xv::bind
{
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

	xc::ArrayPtr array_from_buffer(py::buffer &buff, const xc::Device &device, bool constant)
	{
		auto buff_info = buff.request();
		uint64_t nbytes = buff_info.size * buff_info.itemsize;
		if (nbytes == 0)
		{
			throw std::invalid_argument("Array cannot be initialized from an empty buffer.");
		}
		if (descriptors_to_dtypes.find(buff_info.format) == descriptors_to_dtypes.end())
		{
			throw std::invalid_argument("Unsupported buffer format: " + buff_info.format);
		}
		if (!is_buff_contiguous(buff_info))
		{
			throw std::invalid_argument("Buffer is not contiguous.");
		}
		uint64_t numel = buff_info.size;
		xc::Shape shape(0, {numel}, {1});
		uint8_t *ptr = static_cast<uint8_t *>(buff_info.ptr);
		xc::Dtype dtype = descriptors_to_dtypes.at(buff_info.format);
		return xc::Array::from_buff(ptr, nbytes, shape, dtype, device, constant);
	}

	py::buffer_info array_to_buffer(xc::Array &arr)
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

	xc::ArrayPtr array_from_numpy(py::array &np_arr, const xc::Device &device, bool constant)
	{
		uint64_t nbytes = np_arr.nbytes();
		if (nbytes == 0)
		{
			throw std::invalid_argument("Array cannot be initialized from an empty numpy array.");
		}
		const std::string dtype_fmt = np_arr.dtype().attr("str").cast<std::string>();
		if (descriptors_to_dtypes.find(dtype_fmt) == descriptors_to_dtypes.end())
		{
			throw std::invalid_argument("Unsupported numpy dtype: " + dtype_fmt);
		}
		std::vector<uint64_t> view;
		std::vector<int64_t> stride;
		for (int i = 0; i < np_arr.ndim(); i++)
		{
			view.push_back(np_arr.shape(i));
			stride.push_back(np_arr.strides(i) / np_arr.itemsize());
		}
		xc::Shape shape(np_arr.offset_at(0), view, stride);
		uint8_t *ptr = ptr = static_cast<uint8_t *>(np_arr.mutable_data());
		xc::Dtype dtype = descriptors_to_dtypes.at(dtype_fmt);
		return xc::Array::from_numpy(ptr, nbytes, shape, dtype, device, constant);
	}

	py::array array_to_numpy(xc::Array &arr)
	{
		// Get shape and strides
		std::vector<py::ssize_t> shape;
		std::vector<py::ssize_t> strides;
		for (size_t i = 0; i < arr.get_ndim(); i++)
		{
			shape.push_back(static_cast<py::ssize_t>(arr.get_view()[i]));
			// Ensure correct stride calculation
			strides.push_back(static_cast<py::ssize_t>(arr.get_stride()[i] * arr.get_itemsize()));
		}
		// Create numpy array with read/write access
		return py::array(
			py::buffer_info(
				arr.get_ptr(),							   // Pointer to data
				arr.get_itemsize(),						   // Size of one element
				dtypes_to_descriptors.at(arr.get_dtype()), // Format descriptor
				arr.get_ndim(),							   // Number of dimensions
				shape,									   // Buffer dimensions
				strides									   // Strides (in bytes)
				));
	}

	std::vector<xc::Range> get_arr_ranges(const xc::Array &arr, const py::object &obj)
	{
		std::vector<xc::Range> ranges;
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
		throw xc::PybindInvalidArgumentType(get_pyclass(obj), "int, slice, list, tuple");
	}

	uint64_t map_idx(int64_t len, int64_t idx)
	{
		// TODO: len should be an unsigned?
		if (idx < -len || idx >= len)
		{
			throw std::out_of_range("Index out of range: " + std::to_string(idx) + " not in [-" + std::to_string(len) + ", " + std::to_string(len) + ")");
		}
		return idx < 0 ? idx + len : idx;
	}

	xc::Range slice_to_range(int64_t len, const py::object &obj)
	{
		if (!py::isinstance<py::slice>(obj))
		{
			throw xc::PybindInvalidArgumentType(get_pyclass(obj), "slice");
		}
		auto slice = obj.cast<py::slice>();
		auto start = slice.attr("start").is_none() ? 0 : map_idx(len, slice.attr("start").cast<int64_t>());
		auto stop = slice.attr("stop").is_none() ? len : map_idx(len, slice.attr("stop").cast<int64_t>());
		auto step = slice.attr("step").is_none() ? 1 : slice.attr("step").cast<int64_t>();
		return xc::Range(start, stop, step);
	}

	xc::ArrayPtr obj_to_arr(const py::object &obj, const xc::Device &device_if_scalar)
	{
		if (py::isinstance<xc::Array>(obj))
		{
			return obj.cast<xc::ArrayPtr>();
		}
		else if (py::isinstance<py::float_>(obj))
		{
			return xc::Array::full(obj.cast<float>(), xc::f32, device_if_scalar);
		}
		else if (py::isinstance<py::int_>(obj))
		{
			return xc::Array::full(obj.cast<int>(), xc::i32, device_if_scalar);
		}
		else if (py::isinstance<py::bool_>(obj))
		{
			return xc::Array::full(obj.cast<int>(), xc::b8, device_if_scalar);
		}
		throw xc::PybindInvalidArgumentType(get_pyclass(obj), "float, int, bool, Array");
	}

	bool is_scalar(const py::object &obj)
	{
		return py::isinstance<py::float_>(obj) || py::isinstance<py::int_>(obj) || py::isinstance<py::bool_>(obj);
	}
}