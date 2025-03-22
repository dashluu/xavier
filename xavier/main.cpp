#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "graph/mtl_graph.h"

using namespace xv::backend::metal;
using namespace xv::core;
using namespace xv::graph;

int main()
{
    std::string lib_path = "backend/metal/kernels.metallib";
    auto ctx = std::make_shared<MTLContext>(lib_path);
    std::vector<uint64_t> view = {1, 2, 3};
    auto x1 = Array::full(view, 7.0f);
    auto x2 = Array::arange(view, 0, 1);
    auto x3 = x1->add(x2);
    auto x4 = x1->mul(x2);
    auto x5 = x3->add(x4);
    auto x6 = x3->mul(x4);
    auto x7 = x5->add(x6);
    // x1 = x1->self_add(x2);
    MTLGraph graph(x7, ctx);
    graph.compile();
    std::cout << "Graph:\n"
              << graph.str() << std::endl;
    std::cout << "Forward:" << std::endl;
    graph.forward();
    std::cout << "x1:\n"
              << x1->str() << std::endl
              << std::endl;
    std::cout << "x2:\n"
              << x2->str() << std::endl
              << std::endl;
    std::cout << "x3:\n"
              << x3->str() << std::endl
              << std::endl;
    std::cout << "x4:\n"
              << x4->str() << std::endl
              << std::endl;
    std::cout << "x5:\n"
              << x5->str() << std::endl
              << std::endl;
    std::cout << "x6:\n"
              << x6->str() << std::endl
              << std::endl;
    std::cout << "x7:\n"
              << x7->str() << std::endl
              << std::endl;
    std::cout << "Backward:" << std::endl;
    graph.backward();
    std::cout << "x3's grad:\n"
              << x3->grad->str() << std::endl
              << std::endl;
    std::cout << "x4's grad:\n"
              << x4->grad->str() << std::endl
              << std::endl;
    std::cout << "x5's grad:\n"
              << x5->grad->str() << std::endl
              << std::endl;
    std::cout << "x6's grad:\n"
              << x6->grad->str() << std::endl
              << std::endl;
    std::cout << "x7's grad:\n"
              << x7->grad->str() << std::endl
              << std::endl;
    return 0;
}