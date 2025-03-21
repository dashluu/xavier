#include "mtl_context.h"

namespace xv::backend::metal
{
    std::vector<std::string> MTLContext::num_unary_ops = {"exp", "log", "neg", "recip", "sq", "sqrt"};
    std::vector<std::string> MTLContext::num_binary_ops = {"add", "sub", "mul", "div", "eq", "neq", "lt", "gt", "leq", "geq"};

    void MTLContext::init_kernels(const std::vector<std::string> &ops, const std::unordered_set<Dtype> &dtypes, bool sparse)
    {
        for (auto op : ops)
        {
            init_kernels(op, dtypes, sparse);
        }
    }

    void MTLContext::init_kernels(const std::string &op, const std::unordered_set<Dtype> &dtypes, bool sparse)
    {
        auto prefix = sparse ? "sparse_" : "";
        for (auto dtype : dtypes)
        {
            auto name = prefix + op + "_" + dtype.get_name();
            auto tmp = NS::String::string(name.c_str(), NS::UTF8StringEncoding);
            auto f = NS::TransferPtr<MTL::Function>(lib->newFunction(tmp));
            // TODO: handle error
            NS::Error *error = nullptr;
            auto state = NS::TransferPtr<MTL::ComputePipelineState>(device->newComputePipelineState(f.get(), &error));
            kernels[name] = std::make_shared<MTLKernel>(state, dtype);
        }
    }

    void MTLContext::init_initializer_kernels()
    {
        init_kernels("full", all_dtypes, false);
        init_kernels("arange", num_dtypes, false);
    }

    void MTLContext::init_unary_kernels()
    {
        init_kernels(num_unary_ops, num_dtypes, true);
        init_kernels(num_unary_ops, num_dtypes, false);
    }

    void MTLContext::init_binary_kernels()
    {
        init_kernels(num_binary_ops, num_dtypes, true);
        init_kernels(num_binary_ops, num_dtypes, false);
        init_kernels("matmul", num_dtypes, false);
    }

    void MTLContext::init_util_kernels()
    {
        init_kernels("copy", all_dtypes, true);
        init_kernels("copy", all_dtypes, false);
    }

    MTLContext::MTLContext(const std::string &lib_path)
    {
        pool = NS::TransferPtr<NS::AutoreleasePool>(NS::AutoreleasePool::alloc()->init());
        device = NS::TransferPtr<MTL::Device>(MTL::CreateSystemDefaultDevice());
        auto path = NS::String::string(lib_path.c_str(), NS::ASCIIStringEncoding);
        auto url = NS::URL::fileURLWithPath(path);
        // TODO: handle error
        NS::Error *error = nullptr;
        lib = NS::TransferPtr<MTL::Library>(device->newLibrary(url, &error));
        cmd_queue = NS::TransferPtr<MTL::CommandQueue>(device->newCommandQueue());
        init_initializer_kernels();
        init_unary_kernels();
        init_binary_kernels();
        init_util_kernels();
    }

    void MTLContext::register_kernel(const std::string &name, std::shared_ptr<MTLKernel> kernel)
    {
        if (kernels.contains(name))
        {
            throw std::invalid_argument("Cannot register existing kernel " + name + ".");
        }
        kernels.insert(std::make_pair(name, kernel));
    }
}