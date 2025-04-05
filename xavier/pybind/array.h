#pragma once

#include "utils.h"

namespace xv::bind
{
	xc::ArrayPtr full(const xc::ShapeView &view, const py::object &c, const xc::Dtype &dtype, const xc::Device &device, bool constant);

	xc::ArrayPtr full_like(xc::ArrayPtr arr, const py::object &c, const xc::Device &device, bool constant);

	xc::ArrayPtr unary(const py::object &py_operand, const std::function<xc::ArrayPtr(xc::ArrayPtr, bool)> &f, bool in_place);

	xc::ArrayPtr m_binary(const py::object &py_lhs, const py::object &py_rhs, const std::function<xc::ArrayPtr(xc::ArrayPtr, xc::ArrayPtr)> &f);

	xc::ArrayPtr binary(xc::ArrayPtr lhs_arr, const py::object &py_rhs, const std::function<xc::ArrayPtr(xc::ArrayPtr, xc::ArrayPtr)> &f);

	xc::ArrayPtr m_reduce(const py::object &py_operand,
						  const std::vector<py::int_> &py_dims,
						  const std::function<xc::ArrayPtr(xc::ArrayPtr, const std::vector<xc::usize> &)> &f);

	xc::ArrayPtr reduce(xc::ArrayPtr operand,
						const std::vector<py::int_> &py_dims,
						const std::function<xc::ArrayPtr(xc::ArrayPtr, const std::vector<xc::usize> &)> &f);

	inline xc::ArrayPtr m_add(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
						{ return lhs->add(rhs); });
	}

	inline xc::ArrayPtr add(xc::ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
					  { return lhs->add(rhs); });
	}

	inline xc::ArrayPtr m_self_add(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
						{ return lhs->self_add(rhs); });
	}

	inline xc::ArrayPtr self_add(xc::ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
					  { return lhs->self_add(rhs); });
	}

	inline xc::ArrayPtr m_sub(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
						{ return lhs->sub(rhs); });
	}

	inline xc::ArrayPtr sub(xc::ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
					  { return lhs->sub(rhs); });
	}

	inline xc::ArrayPtr m_self_sub(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
						{ return lhs->self_sub(rhs); });
	}

	inline xc::ArrayPtr self_sub(xc::ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
					  { return lhs->self_sub(rhs); });
	}

	inline xc::ArrayPtr m_mul(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
						{ return lhs->mul(rhs); });
	}

	inline xc::ArrayPtr mul(xc::ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
					  { return lhs->mul(rhs); });
	}

	inline xc::ArrayPtr m_self_mul(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
						{ return lhs->self_mul(rhs); });
	}

	inline xc::ArrayPtr self_mul(xc::ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
					  { return lhs->self_mul(rhs); });
	}

	inline xc::ArrayPtr m_div(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
						{ return lhs->div(rhs); });
	}

	inline xc::ArrayPtr div(xc::ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
					  { return lhs->div(rhs); });
	}

	inline xc::ArrayPtr m_self_div(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
						{ return lhs->self_div(rhs); });
	}

	inline xc::ArrayPtr self_div(xc::ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
					  { return lhs->self_div(rhs); });
	}

	inline xc::ArrayPtr m_matmul(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
						{ return lhs->matmul(rhs); });
	}

	inline xc::ArrayPtr matmul(xc::ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
					  { return lhs->matmul(rhs); });
	}

	inline xc::ArrayPtr m_eq(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
						{ return lhs->eq(rhs); });
	}

	inline xc::ArrayPtr eq(xc::ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
					  { return lhs->eq(rhs); });
	}

	inline xc::ArrayPtr m_neq(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
						{ return lhs->neq(rhs); });
	}

	inline xc::ArrayPtr neq(xc::ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
					  { return lhs->neq(rhs); });
	}

	inline xc::ArrayPtr m_lt(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
						{ return lhs->lt(rhs); });
	}

	inline xc::ArrayPtr lt(xc::ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
					  { return lhs->lt(rhs); });
	}

	inline xc::ArrayPtr m_gt(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
						{ return lhs->gt(rhs); });
	}

	inline xc::ArrayPtr gt(xc::ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
					  { return lhs->gt(rhs); });
	}

	inline xc::ArrayPtr m_leq(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
						{ return lhs->leq(rhs); });
	}

	inline xc::ArrayPtr leq(xc::ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
					  { return lhs->leq(rhs); });
	}

	inline xc::ArrayPtr m_geq(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
						{ return lhs->geq(rhs); });
	}

	inline xc::ArrayPtr geq(xc::ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](xc::ArrayPtr lhs, xc::ArrayPtr rhs)
					  { return lhs->geq(rhs); });
	}

	inline xc::ArrayPtr m_exp(const py::object &operand, bool in_place)
	{
		return unary(operand, [](xc::ArrayPtr arr, bool in_place)
					 { return arr->exp(in_place); }, in_place);
	}

	inline xc::ArrayPtr m_log(const py::object &operand, bool in_place)
	{
		return unary(operand, [](xc::ArrayPtr arr, bool in_place)
					 { return arr->log(in_place); }, in_place);
	}

	inline xc::ArrayPtr m_neg(const py::object &operand, bool in_place)
	{
		return unary(operand, [](xc::ArrayPtr arr, bool in_place)
					 { return arr->neg(in_place); }, in_place);
	}

	inline xc::ArrayPtr m_identity(const py::object &operand)
	{
		return unary(operand, [](xc::ArrayPtr arr, bool in_place)
					 { return arr->identity(); }, false);
	}

	inline xc::ArrayPtr m_recip(const py::object &operand, bool in_place)
	{
		return unary(operand, [](xc::ArrayPtr arr, bool in_place)
					 { return arr->recip(in_place); }, in_place);
	}

	inline xc::ArrayPtr m_sq(const py::object &operand, bool in_place)
	{
		return unary(operand, [](xc::ArrayPtr arr, bool in_place)
					 { return arr->sq(in_place); }, in_place);
	}

	inline xc::ArrayPtr m_sqrt(const py::object &operand, bool in_place)
	{
		return unary(operand, [](xc::ArrayPtr arr, bool in_place)
					 { return arr->sqrt(in_place); }, in_place);
	}

	inline xc::ArrayPtr permute(xc::ArrayPtr operand, const xc::ShapeOrder &order)
	{
		return operand->permute(order);
	}

	inline xc::ArrayPtr m_permute(const py::object &operand, const xc::ShapeOrder &order)
	{
		return permute(obj_to_arr(operand, xc::device0), order);
	}

	inline xc::ArrayPtr T(xc::ArrayPtr operand, xc::isize start_dim, xc::isize end_dim)
	{
		return operand->T(map_idx(operand->get_ndim(), start_dim), map_idx(operand->get_ndim(), end_dim));
	}

	inline xc::ArrayPtr m_T(const py::object &operand, xc::isize start_dim, xc::isize end_dim)
	{
		return T(obj_to_arr(operand, xc::device0), start_dim, end_dim);
	}

	inline xc::ArrayPtr flatten(xc::ArrayPtr operand, xc::isize start_dim, xc::isize end_dim)
	{
		return operand->flatten(map_idx(operand->get_ndim(), start_dim), map_idx(operand->get_ndim(), end_dim));
	}

	inline xc::ArrayPtr m_flatten(const py::object &operand, xc::isize start_dim, xc::isize end_dim)
	{
		return flatten(obj_to_arr(operand, xc::device0), start_dim, end_dim);
	}

	inline xc::ArrayPtr sum(xc::ArrayPtr operand, const std::vector<py::int_> &dims)
	{
		return reduce(operand, dims, [](xc::ArrayPtr arr, const std::vector<xc::usize> &dims)
					  { return arr->sum(dims); });
	}

	inline xc::ArrayPtr m_sum(const py::object &operand, const std::vector<py::int_> &dims)
	{
		return m_reduce(operand, dims, [](xc::ArrayPtr arr, const std::vector<xc::usize> &dims)
						{ return arr->sum(dims); });
	}

	inline xc::ArrayPtr max(xc::ArrayPtr operand, const std::vector<py::int_> &dims)
	{
		return reduce(operand, dims, [](xc::ArrayPtr arr, const std::vector<xc::usize> &dims)
					  { return arr->max(dims); });
	}

	inline xc::ArrayPtr m_max(const py::object &operand, const std::vector<py::int_> &dims)
	{
		return m_reduce(operand, dims, [](xc::ArrayPtr arr, const std::vector<xc::usize> &dims)
						{ return arr->max(dims); });
	}
}