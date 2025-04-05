#pragma once

#include "mtl_kernel.h"

namespace xv::backend::metal
{
    class MTLContext : public std::enable_shared_from_this<MTLContext>
    {
    private:
        static std::vector<std::string> numeric_unary;
        static std::vector<std::string> numeric_binary;
        static std::vector<std::string> cmp_all;
        static std::vector<std::string> numeric_reduction;
        NS::SharedPtr<MTL::Device> device;
        NS::SharedPtr<MTL::Library> lib;
        NS::SharedPtr<MTL::CommandQueue> cmd_queue;
        std::unordered_map<std::string, std::shared_ptr<MTLKernel>> kernels;

        void init_kernel(const std::string &name, const Dtype &dtype);
        void init_kernels(const std::vector<std::string> &ops, const std::unordered_set<Dtype> &dtypes, const std::vector<std::string> &modes);
        void init_kernels(const std::string &op, const std::unordered_set<Dtype> &dtypes, const std::vector<std::string> &modes);
        void init_kernels(const std::string &op, const std::unordered_set<Dtype> &dtypes);
        void init_initializer_kernels();
        void init_unary_kernels();
        void init_binary_kernels();
        void init_reduction_kernels();

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