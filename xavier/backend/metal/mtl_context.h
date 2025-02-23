#pragma once

#include "mtl_kernel.h"

namespace xv::backend::metal
{
    class MTLContext
    {
    private:
        static std::vector<std::string> initializer_ops;
        static std::vector<std::string> unary_ops;
        static std::vector<std::string> binary_ops;
        static std::vector<std::string> util_ops;
        NS::SharedPtr<MTL::Device> device;
        NS::SharedPtr<MTL::Library> lib;
        NS::SharedPtr<MTL::CommandQueue> cmd_queue;
        std::unordered_map<std::string, std::shared_ptr<MTLKernel>> kernels;

        void init_kernels(std::vector<std::string> &ops, bool sparse);

    public:
        MTLContext(const std::string &lib_path);

        void register_kernel(const std::string &name, std::shared_ptr<MTLKernel> kernel);

        std::shared_ptr<MTLKernel> get_kernel(const std::string &name)
        {
            return kernels[name];
        }

        NS::SharedPtr<MTL::Device> get_device()
        {
            return device;
        }

        NS::SharedPtr<MTL::CommandQueue> get_cmd_queue()
        {
            return cmd_queue;
        }
    };
}