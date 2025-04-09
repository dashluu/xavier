#include "mtl_context.h"

namespace xv::backend::metal
{
    std::vector<std::string> MTLContext::numeric_unary = {"identity", "exp", "log", "neg", "recip", "sq", "sqrt"};
    std::vector<std::string> MTLContext::numeric_binary = {"add", "sub", "mul", "div", "lt", "gt", "leq", "geq"};
    std::vector<std::string> MTLContext::cmp_all = {"eq", "neq"};
    std::vector<std::string> MTLContext::numeric_reduction = {"sum", "max"};

    void MTLContext::init_kernel(const std::string &name, const Dtype &dtype)
    {
        auto kernel = std::make_shared<MTLKernel>(name, dtype);
        kernel->init(device, lib);
        kernels[name] = kernel;
    }

    void MTLContext::init_kernels(const std::vector<std::string> &ops, const std::unordered_set<Dtype> &dtypes, const std::vector<std::string> &modes)
    {
        for (auto &op : ops)
        {
            init_kernels(op, dtypes, modes);
        }
    }

    void MTLContext::init_kernels(const std::string &op, const std::unordered_set<Dtype> &dtypes, const std::vector<std::string> &modes)
    {
        for (auto &mode : modes)
        {
            for (auto &dtype : dtypes)
            {
                auto name = op + "_" + mode + "_" + dtype.get_name();
                init_kernel(name, dtype);
            }
        }
    }

    void MTLContext::init_kernels(const std::string &op, const std::unordered_set<Dtype> &dtypes)
    {
        for (auto &dtype : dtypes)
        {
            auto name = op + "_" + dtype.get_name();
            init_kernel(name, dtype);
        }
    }

    void MTLContext::init_initializer_kernels()
    {
        init_kernels("full", all_dtypes);
        init_kernels("arange", numeric_dtypes);
    }

    void MTLContext::init_unary_kernels()
    {
        init_kernels(numeric_unary, numeric_dtypes, {"vv", "sv", "vs", "ss"});
    }

    void MTLContext::init_binary_kernels()
    {
        init_kernels(numeric_binary, numeric_dtypes, {"vv", "sv", "vs", "ss"});
        init_kernels(cmp_all, all_dtypes, {"vv", "sv", "vs", "ss"});
        init_kernels("matmul", numeric_dtypes, {"vv", "vs"});
    }

    void MTLContext::init_reduction_kernels()
    {
        init_kernels("reduce_col_copy", numeric_dtypes, {"vv"});
        for (auto &op : numeric_reduction)
        {
            init_kernels(op + "_all", numeric_dtypes, {"vv", "vs"});
            init_kernels(op + "_col", numeric_dtypes);
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