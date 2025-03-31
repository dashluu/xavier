#pragma once

#include "utils.h"

namespace xv::bind
{
	using namespace xc;

	ArrayPtr full(const std::vector<uint64_t> &view, const py::object &c, const Dtype &dtype, const Device &device, bool constant);

	ArrayPtr full_like(ArrayPtr arr, const py::object &c, const Device &device, bool constant);

	ArrayPtr unary(const py::object &operand_obj, const std::function<ArrayPtr(ArrayPtr, bool)> &f, bool in_place);

	ArrayPtr m_binary(const py::object &lhs_obj, const py::object &rhs_obj, const std::function<ArrayPtr(ArrayPtr, ArrayPtr)> &f);

	ArrayPtr binary(ArrayPtr lhs_arr, const py::object &rhs_obj, const std::function<ArrayPtr(ArrayPtr, ArrayPtr)> &f);

	ArrayPtr reduce(const py::object &operand_obj, const std::function<ArrayPtr(ArrayPtr)> &f);

	inline ArrayPtr m_add(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
						{ return lhs->add(rhs); });
	}

	inline ArrayPtr add(ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
					  { return lhs->add(rhs); });
	}

	inline ArrayPtr m_self_add(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
						{ return lhs->self_add(rhs); });
	}

	inline ArrayPtr self_add(ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
					  { return lhs->self_add(rhs); });
	}

	inline ArrayPtr m_sub(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
						{ return lhs->sub(rhs); });
	}

	inline ArrayPtr sub(ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
					  { return lhs->sub(rhs); });
	}

	inline ArrayPtr m_self_sub(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
						{ return lhs->self_sub(rhs); });
	}

	inline ArrayPtr self_sub(ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
					  { return lhs->self_sub(rhs); });
	}

	inline ArrayPtr m_mul(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
						{ return lhs->mul(rhs); });
	}

	inline ArrayPtr mul(ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
					  { return lhs->mul(rhs); });
	}

	inline ArrayPtr m_self_mul(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
						{ return lhs->self_mul(rhs); });
	}

	inline ArrayPtr self_mul(ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
					  { return lhs->self_mul(rhs); });
	}

	inline ArrayPtr m_div(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
						{ return lhs->div(rhs); });
	}

	inline ArrayPtr div(ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
					  { return lhs->div(rhs); });
	}

	inline ArrayPtr m_self_div(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
						{ return lhs->self_div(rhs); });
	}

	inline ArrayPtr self_div(ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
					  { return lhs->self_div(rhs); });
	}

	inline ArrayPtr m_matmul(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
						{ return lhs->matmul(rhs); });
	}

	inline ArrayPtr matmul(ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
					  { return lhs->matmul(rhs); });
	}

	inline ArrayPtr m_eq(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
						{ return lhs->eq(rhs); });
	}

	inline ArrayPtr eq(ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
					  { return lhs->eq(rhs); });
	}

	inline ArrayPtr m_neq(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
						{ return lhs->neq(rhs); });
	}

	inline ArrayPtr neq(ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
					  { return lhs->neq(rhs); });
	}

	inline ArrayPtr m_lt(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
						{ return lhs->lt(rhs); });
	}

	inline ArrayPtr lt(ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
					  { return lhs->lt(rhs); });
	}

	inline ArrayPtr m_gt(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
						{ return lhs->gt(rhs); });
	}

	inline ArrayPtr gt(ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
					  { return lhs->gt(rhs); });
	}

	inline ArrayPtr m_leq(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
						{ return lhs->leq(rhs); });
	}

	inline ArrayPtr leq(ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
					  { return lhs->leq(rhs); });
	}

	inline ArrayPtr m_geq(const py::object &lhs, const py::object &rhs)
	{
		return m_binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
						{ return lhs->geq(rhs); });
	}

	inline ArrayPtr geq(ArrayPtr lhs, const py::object &rhs)
	{
		return binary(lhs, rhs, [](ArrayPtr lhs, ArrayPtr rhs)
					  { return lhs->geq(rhs); });
	}

	inline ArrayPtr exp(const py::object &operand, bool in_place)
	{
		return unary(operand, [](ArrayPtr arr, bool in_place)
					 { return arr->exp(in_place); }, in_place);
	}

	inline ArrayPtr log(const py::object &operand, bool in_place)
	{
		return unary(operand, [](ArrayPtr arr, bool in_place)
					 { return arr->log(in_place); }, in_place);
	}

	inline ArrayPtr neg(const py::object &operand, bool in_place)
	{
		return unary(operand, [](ArrayPtr arr, bool in_place)
					 { return arr->neg(in_place); }, in_place);
	}

	inline ArrayPtr recip(const py::object &operand, bool in_place)
	{
		return unary(operand, [](ArrayPtr arr, bool in_place)
					 { return arr->recip(in_place); }, in_place);
	}

	inline ArrayPtr sq(const py::object &operand, bool in_place)
	{
		return unary(operand, [](ArrayPtr arr, bool in_place)
					 { return arr->sq(in_place); }, in_place);
	}

	inline ArrayPtr sqrt(const py::object &operand, bool in_place)
	{
		return unary(operand, [](ArrayPtr arr, bool in_place)
					 { return arr->sqrt(in_place); }, in_place);
	}

	inline ArrayPtr permute(ArrayPtr operand, const std::vector<uint64_t> &order)
	{
		return operand->permute(order);
	}

	inline ArrayPtr m_permute(const py::object &operand, const std::vector<uint64_t> &order)
	{
		return permute(obj_to_arr(operand, device0), order);
	}

	inline ArrayPtr T(ArrayPtr operand, int64_t start_dim, int64_t end_dim)
	{
		return operand->T(map_idx(operand->get_ndim(), start_dim), map_idx(operand->get_ndim(), end_dim));
	}

	inline ArrayPtr m_T(const py::object &operand, int64_t start_dim, int64_t end_dim)
	{
		return T(obj_to_arr(operand, device0), start_dim, end_dim);
	}

	inline ArrayPtr flatten(ArrayPtr operand, int64_t start_dim, int64_t end_dim)
	{
		return operand->flatten(map_idx(operand->get_ndim(), start_dim), map_idx(operand->get_ndim(), end_dim));
	}

	inline ArrayPtr m_flatten(const py::object &operand, int64_t start_dim, int64_t end_dim)
	{
		return flatten(obj_to_arr(operand, device0), start_dim, end_dim);
	}

	inline ArrayPtr sum(const py::object &operand)
	{
		return reduce(operand, [](ArrayPtr arr)
					  { return arr->sum(); });
	}
}