#include "mtl_context.h"

namespace xv::backend::metal
{
    std::vector<std::string> MTLContext::numeric_unary_ops = {"identity", "exp", "log", "neg", "recip", "sq", "sqrt"};
    std::vector<std::string> MTLContext::numeric_binary_ops = {"add", "sub", "mul", "div", "lt", "gt", "leq", "geq"};
    std::vector<std::string> MTLContext::cmp_ops = {"eq", "neq"};
    std::vector<std::string> MTLContext::numeric_reduction_ops = {"sum", "max"};

    void MTLContext::init_kernels(const std::vector<std::string> &ops, const std::unordered_set<Dtype> &dtypes, const std::string &mode)
    {
        for (auto op : ops)
        {
            init_kernels(op, dtypes, mode);
        }
    }

    void MTLContext::init_kernels(const std::string &op, const std::unordered_set<Dtype> &dtypes, const std::string &mode)
    {
        for (auto dtype : dtypes)
        {
            auto name = op + "_" + (mode.empty() ? "" : mode + "_") + dtype.get_name();
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
        init_kernels("full", all_dtypes);
        init_kernels("arange", numeric_dtypes);
    }

    void MTLContext::init_unary_kernels()
    {
        init_kernels(numeric_unary_ops, numeric_dtypes, "vv");
        init_kernels(numeric_unary_ops, numeric_dtypes, "sv");
        init_kernels(numeric_unary_ops, numeric_dtypes, "vs");
        init_kernels(numeric_unary_ops, numeric_dtypes, "ss");
    }

    void MTLContext::init_binary_kernels()
    {
        init_kernels(numeric_binary_ops, numeric_dtypes, "vv");
        init_kernels(numeric_binary_ops, numeric_dtypes, "sv");
        init_kernels(numeric_binary_ops, numeric_dtypes, "vs");
        init_kernels(numeric_binary_ops, numeric_dtypes, "ss");
        init_kernels(cmp_ops, all_dtypes, "vv");
        init_kernels(cmp_ops, all_dtypes, "sv");
        init_kernels(cmp_ops, all_dtypes, "vs");
        init_kernels(cmp_ops, all_dtypes, "ss");
        init_kernels("matmul", numeric_dtypes, "vv");
        init_kernels("matmul", numeric_dtypes, "vs");
    }

    void MTLContext::init_reduction_kernels()
    {
        for (auto op : numeric_reduction_ops)
        {
            init_kernels(op + "_all", numeric_dtypes, "vv");
            init_kernels(op + "_all", numeric_dtypes, "vs");
            init_kernels(op + "_col", numeric_dtypes, "vv");
        }
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
        // Initializes kernels here
        init_initializer_kernels();
        init_unary_kernels();
        init_binary_kernels();
        init_reduction_kernels();
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