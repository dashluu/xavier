#pragma once

#include "../common.h"

namespace xv::core
{
    class Shape : public IStr
    {
    private:
        uint64_t offset;
        std::vector<uint64_t> view;
        std::vector<int64_t> stride;

        void check_ranges(const std::vector<Range> &ranges) const
        {
            if (ranges.size() != get_ndim())
            {
                throw std::invalid_argument("The number of ranges does not match the number of dimensions: " +
                                            std::to_string(ranges.size()) + " and " +
                                            std::to_string(get_ndim()) + ".");
            }
            for (int i = 0; i < ranges.size(); i++)
            {
                auto &range = ranges[i];
                if (range.start >= view[i])
                {
                    throw std::invalid_argument("Invalid starting point for range: " + std::to_string(range.start));
                }
                if (range.stop > view[i])
                {
                    throw std::invalid_argument("Invalid stopping point for range: " + std::to_string(range.stop));
                }
                if (range.step == 0)
                {
                    throw std::invalid_argument("Step cannot be zero.");
                }
                if (range.start < range.stop && range.step < 0)
                {
                    throw std::invalid_argument("Step must be positive if start < stop: " + std::to_string(range.start) + " < " + std::to_string(range.stop));
                }
                if (range.start > range.stop && range.step > 0)
                {
                    throw std::invalid_argument("Step must be negative if start > stop: " + std::to_string(range.start) + " > " + std::to_string(range.stop));
                }
            }
        }

        void check_permute(const std::vector<uint64_t> &order) const
        {
            auto n = view.size();
            if (order.size() != n)
            {
                throw std::invalid_argument("The number of dimensions in the order does not match the number of dimensions in the shape: " +
                                            std::to_string(order.size()) + " and " + std::to_string(n) + ".");
            }
            std::vector<bool> flag(n, false);
            for (auto o : order)
            {
                if (o >= n)
                {
                    throw std::invalid_argument("The order must be a permutation of the dimensions but got " + vnumstr(order) + ".");
                }
                flag[o] = true;
            }
            for (auto f : flag)
            {
                if (!f)
                {
                    throw std::invalid_argument("The order must be a permutation of the dimensions but got " + vnumstr(order) + ".");
                }
            }
        }

    public:
        Shape() : Shape(0, {1}, {1}) {}

        Shape(uint64_t offset, const std::vector<uint64_t> &view, const std::vector<int64_t> &stride)
        {
            check_view(view);
            if (view.size() != stride.size())
            {
                throw std::invalid_argument("View and stride do not have the same number of dimensions: " +
                                            std::to_string(view.size()) + " and " + std::to_string(stride.size()) + ".");
            }
            this->offset = offset;
            this->view = view;
            this->stride = stride;
        }

        Shape(uint64_t offset, const std::vector<uint64_t> &view)
        {
            check_view(view);
            this->offset = offset;
            this->view = view;
            stride.resize(view.size());
            uint64_t s = 1;
            for (int i = view.size() - 1; i >= 0; i--)
            {
                stride[i] = s;
                s *= view[i];
            }
        }

        Shape(const std::vector<uint64_t> &view) : Shape(0, view) {}

        Shape(const Shape &shape) : Shape(shape.offset, shape.view, shape.stride) {}

        Shape &operator=(const Shape &shape)
        {
            offset = shape.offset;
            view = shape.view;
            stride = shape.stride;
            return *this;
        }

        uint64_t get_offset() const { return offset; }

        const std::vector<uint64_t> &get_view() const { return view; }

        const std::vector<int64_t> &get_stride() const { return stride; }

        bool is_contiguous() const { return stride == get_contiguous_stride(); }

        static void check_view(const std::vector<uint64_t> &view)
        {
            if (view.size() == 0)
            {
                throw std::invalid_argument("Shape must have at least one dimension.");
            }
            if (std::any_of(view.begin(), view.end(), [](uint64_t v)
                            { return v == 0; }))
            {
                throw std::invalid_argument("Dimension cannot be zero.");
            }
        }

        std::vector<int64_t> get_contiguous_stride() const
        {
            std::vector<int64_t> contiguous_stride(stride.size(), 0);
            int64_t s = 1;
            for (int i = view.size() - 1; i >= 0; i--)
            {
                contiguous_stride[i] = s;
                s *= view[i];
            }
            return contiguous_stride;
        }

        std::vector<uint64_t> get_elms_per_dim() const
        {
            std::vector<uint64_t> elms_per_dim(view.size(), 0);
            uint64_t n = 1;
            for (int i = view.size() - 1; i >= 0; i--)
            {
                n *= view[i];
                elms_per_dim[i] = n;
            }
            return elms_per_dim;
        }

        uint64_t get_ndim() const { return view.size(); }

        uint64_t get_numel() const { return std::accumulate(view.begin(), view.end(), 1, std::multiplies<uint64_t>()); }

        bool broadcastable(const std::vector<uint64_t> &rhs) const
        {
            if (view == rhs)
            {
                return true;
            }
            for (auto view_iter = view.rbegin(), rhs_iter = rhs.rbegin();
                 view_iter != view.rend() && rhs_iter != rhs.rend();
                 view_iter++, rhs_iter++)
            {
                if (*view_iter != *rhs_iter && *view_iter != 1 && *rhs_iter != 1)
                {
                    return false;
                }
            }
            return true;
        }

        // One-direction broadcast check
        bool broadcastable_to(const std::vector<uint64_t> &target) const
        {
            if (view == target)
            {
                return true;
            }
            if (view.size() > target.size())
            {
                return false;
            }
            for (auto view_iter = view.rbegin(), target_iter = target.rbegin(); view_iter != view.rend(); view_iter++, target_iter++)
            {
                if (*view_iter != *target_iter && *view_iter != 1)
                {
                    return false;
                }
            }
            return true;
        }

        bool matmul_broadcastable(const std::vector<uint64_t> &rhs) const
        {
            if (view.size() < 2 || view[view.size() - 1] != rhs[rhs.size() - 2])
            {
                return false;
            }
            for (auto view_iter = view.begin(), rhs_iter = rhs.begin();
                 view_iter != view.end() - 2 && rhs_iter != rhs.end() - 2;
                 view_iter++, rhs_iter++)
            {
                if (*view_iter != *rhs_iter && *view_iter != 1 && *rhs_iter != 1)
                {
                    return false;
                }
            }
            return true;
        }

        // One-direction broadcast
        Shape broadcast_to(const std::vector<uint64_t> &target) const
        {
            if (view == target)
            {
                return *this;
            }
            if (!broadcastable_to(target))
            {
                throw std::invalid_argument("Cannot broadcast shape (" + vnumstr(view) + ") to (" + vnumstr(target) + ").");
            }
            auto v = view;
            auto diff = target.size() - v.size();
            v.insert(v.begin(), diff, 1);
            auto s = Shape(offset, v);
            std::fill_n(s.stride.begin(), diff, 0);
            for (int i = 0; i < target.size(); i++)
            {
                if (v[i] < target[i])
                {
                    s.view[i] = target[i];
                    s.stride[i] = 0;
                }
            }
            return s;
        }

        Shape broadcast(const std::vector<uint64_t> &rhs) const
        {
            if (view == rhs)
            {
                return *this;
            }
            if (!broadcastable(rhs))
            {
                throw std::invalid_argument("Cannot broadcast shape (" + vnumstr(view) + ") and (" + vnumstr(rhs) + ").");
            }
            auto v1 = view;
            auto v2 = rhs;
            auto ndim = std::max(v1.size(), v2.size());
            auto diff1 = ndim - v1.size();
            auto diff2 = ndim - v2.size();
            v1.insert(v1.begin(), diff1, 1);
            v2.insert(v2.begin(), diff2, 1);
            auto s = Shape(offset, v1);
            std::fill_n(s.stride.begin(), diff1, 0);
            for (int i = 0; i < ndim; i++)
            {
                if (v1[i] < v2[i])
                {
                    s.view[i] = v2[i];
                    s.stride[i] = 0;
                }
            }
            return s;
        }

        Shape reshape(const std::vector<uint64_t> &target)
        {
            // TODO: fix this
            return Shape(offset, target);
        }

        Shape remove(uint64_t dim) const
        {
            auto v = view;
            auto s = stride;
            v.erase(v.begin() + dim);
            s.erase(s.begin() + dim);
            return Shape(offset, v, s);
        }

        Shape permute(const std::vector<uint64_t> &order) const
        {
            check_permute(order);
            auto n = view.size();
            std::vector<uint64_t> v(n, 0);
            std::vector<int64_t> s(n, 0);
            for (int i = 0; i < n; i++)
            {
                v[i] = view[order[i]];
                s[i] = stride[order[i]];
            }
            return Shape(offset, v, s);
        }

        std::vector<uint64_t> undo_permute_view(const std::vector<uint64_t> &order) const
        {
            /*
            Example 1:
            2, 2, 4, 3, 5
            1, 2, 0, 4, 3
            2, 4, 2, 5, 3
            2, 0, 1, 4, 3
            2, 2, 4, 3, 5

            Example 2:
            2 3 4
            2 0 1
            4 2 3
            1 2 0
            2 3 4

            Example 3:
            2 3 4 5
            1 3 2 0
            3 5 4 2
            3 0 2 1
            2 3 4 5
            */
            check_permute(order);
            std::vector<uint64_t> reversed_order(order.size());
            for (int i = 0; i < order.size(); i++)
            {
                reversed_order[order[i]] = i;
            }
            return reversed_order;
        }

        Shape undo_permute(const std::vector<uint64_t> &order) const
        {
            return permute(undo_permute_view(order));
        }

        Shape slice(const std::vector<Range> &ranges) const
        {
            check_ranges(ranges);
            auto o = offset;
            for (int i = 0; i < ranges.size(); i++)
            {
                o += ranges[i].start * stride[i];
            }
            std::vector<uint64_t> v(get_ndim());
            std::vector<int64_t> s(get_ndim());
            for (int i = 0; i < ranges.size(); i++)
            {
                auto &range = ranges[i];
                auto d = range.start <= range.stop ? range.stop - range.start : range.start - range.stop;
                v[i] = static_cast<uint64_t>(ceil(static_cast<double>(d) / std::abs(range.step)));
                s[i] = stride[i] * range.step;
            }
            return Shape(o, v, s);
        }

        bool operator==(const Shape &shape) const { return view == shape.view; }

        bool operator!=(const Shape &shape) const { return !(*this == shape); }

        uint64_t operator[](uint64_t dim) const { return view[dim]; }

        const std::string str() const override
        {
            return "(" + vnumstr(view) + ")";
        }

        std::vector<uint64_t>::const_iterator cbegin() const
        {
            return view.cbegin();
        }

        std::vector<uint64_t>::const_iterator cend() const
        {
            return view.cend();
        }

        std::vector<uint64_t>::const_reverse_iterator crbegin() const
        {
            return view.crbegin();
        }

        std::vector<uint64_t>::const_reverse_iterator crend() const
        {
            return view.crend();
        }
    };
}