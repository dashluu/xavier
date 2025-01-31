#pragma once

#include "array.h"

namespace xv::core
{
    class Graph : public IStr
    {
    protected:
        std::shared_ptr<Array> root;

        const std::string recur_str(std::shared_ptr<Array> arr, std::unordered_set<IdType> &visited) const;

    public:
        Graph(std::shared_ptr<Array> root) : root(root) {}

        Graph(const Graph &) = delete;

        Graph &operator=(const Graph &) = delete;

        std::shared_ptr<Array> get_root() { return root; }

        const std::string str() const override;

        virtual void forward() {}

        virtual void backward() {}
    };
}