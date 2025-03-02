#pragma once

#include "../core/array.h"

namespace xv::graph
{
    using namespace xv::core;

    class Graph : public IStr
    {
    protected:
        std::shared_ptr<Array> root;

    public:
        Graph(std::shared_ptr<Array> root) : root(root) {}

        Graph(const Graph &) = delete;

        Graph &operator=(const Graph &) = delete;

        std::shared_ptr<Array> get_root() { return root; }

        virtual void compile() = 0;

        virtual void forward() = 0;

        virtual void backward() = 0;
    };
}