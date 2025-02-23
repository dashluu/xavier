#include "mtl_context.h"

namespace xv::backend::metal
{
    std::vector<std::string> MTLContext::initializer_ops = {"constant_c", "arange"};
    std::vector<std::string> MTLContext::unary_ops = {"exp", "log", "neg", "recip"};
    std::vector<std::string> MTLContext::binary_ops = {"add", "sub", "mul", "div", "eq", "neq", "lt", "gt", "leq", "geq"};
    std::vector<std::string> MTLContext::util_ops = {"copy"};

    void MTLContext::init_kernels(std::vector<std::string> &ops, bool sparse)
    {
        auto prefix = sparse ? "sparse_" : "";
        for (auto op : ops)
        {
            for (auto dtype : num_dtypes)
            {
                auto name = prefix + op + "_" + dtype.get_name();
                auto tmp = NS::String::string(name.c_str(), NS::UTF8StringEncoding);
                auto f = lib->newFunction(tmp);
                kernels[name] = std::make_shared<MTLKernel>(device, f, dtype);
            }
        }
    }

    MTLContext::MTLContext(const std::string &lib_path)
    {
        device = NS::TransferPtr<MTL::Device>(MTL::CreateSystemDefaultDevice());
        auto path = NS::String::string(lib_path.c_str(), NS::ASCIIStringEncoding);
        auto url = NS::URL::fileURLWithPath(path);
        // TODO: handle error
        NS::Error *error = nullptr;
        lib = NS::TransferPtr<MTL::Library>(device->newLibrary(url, &error));
        cmd_queue = NS::TransferPtr<MTL::CommandQueue>(device->newCommandQueue());
        init_kernels(initializer_ops, false);
        init_kernels(unary_ops, false);
        init_kernels(unary_ops, true);
        init_kernels(binary_ops, false);
        init_kernels(binary_ops, true);
        init_kernels(util_ops, false);
        init_kernels(util_ops, true);
    }

    void MTLContext::register_kernel(const std::string &name, std::shared_ptr<MTLKernel> kernel)
    {
        if (kernels.contains(name))
        {
            throw std::runtime_error("Cannot register existing kernel " + name + ".");
        }
        kernels.insert(std::make_pair(name, kernel));
    }
}