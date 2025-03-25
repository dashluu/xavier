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
    std::vector<uint64_t> view1 = {2, 2, 3};
    std::vector<uint64_t> view2 = {1, 3, 4};
    auto x1 = Array::full(view1, 7.0f);
    auto x2 = Array::arange(view2, 0, 1);
    auto x3 = x1->matmul(x2);
    MTLGraph graph(x3, ctx);
    graph.compile();
    std::cout << "Graph:\n"
              << graph.str() << std::endl;
    std::cout << "Forward:" << std::endl;
    graph.forward();
    std::cout << x3->get_shape().str() << std::endl;
    std::cout << "x3:\n"
              << x3->str() << std::endl
              << std::endl;
    return 0;
}