#pragma once

#include "../core/array.h"

namespace xv::graph
{
    using namespace xv::core;

    class Graph : public IStr
    {
    protected:
        ArrayPtr root;

    public:
        Graph(ArrayPtr root) : root(root) {}

        Graph(const Graph &) = delete;

        Graph &operator=(const Graph &) = delete;

        ArrayPtr get_root() { return root; }

        virtual void compile() = 0;

        virtual void forward() = 0;

        virtual void backward() = 0;
    };
}