#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "backend/metal/mtl_kernels.h"
#include "core/array.h"

using namespace xv::backend::metal;
using namespace xv::core;

int main()
{
    auto lib_path = "backend/metal/kernels.metallib";
    MTLContext ctx(lib_path);
    std::vector<uint64_t> view = {1, 2, 3};
    auto x1 = Array::constant(view, 7.0f);
    x1->alloc();
    xv::backend::metal::constant(x1, 7.0f, ctx);
    auto x2 = Array::arange(view, 0, 1);
    x2->alloc();
    xv::backend::metal::arange(x2, 0, 1, ctx);
    auto x3 = Array::constant(view, 7.0f);
    x3->alloc();
    xv::backend::metal::constant(x3, 7.0f, ctx);
    std::cout << x1->str() << std::endl;
    std::cout << x2->str() << std::endl;
    std::cout << x3->str() << std::endl;
    xv::backend::metal::copy(x2, x3, ctx);
    std::cout << x3->str() << std::endl;
    return 0;
}