#include <stdexcept>
#include <string>

namespace xv::core
{
    class MTLGraphNotCompiledException : public std::runtime_error
    {
    public:
        MTLGraphNotCompiledException() : std::runtime_error("Graph has not been compiled.") {}
    };

    class IncompatShapesForOp : public std::invalid_argument
    {
    public:
        IncompatShapesForOp(const std::string &op, const std::string lhs_view, const std::string rhs_view) : std::invalid_argument("Cannot run operator " + op + " on incompatible shapes " + lhs_view + " and " + rhs_view + ".") {}
    };

    class IncompatDtypesForOp : public std::invalid_argument
    {
    public:
        IncompatDtypesForOp(const std::string &op, const std::string lhs_dtype, const std::string rhs_dtype) : std::invalid_argument("Cannot run operator " + op + " on incompatible data types " + lhs_dtype + " and " + rhs_dtype + ".") {}
    };

    class IncompatDtypeForOp : public std::invalid_argument
    {
    public:
        IncompatDtypeForOp(const std::string &op, const std::string dtype) : std::invalid_argument("Cannot run operator " + op + " on incompatible data type " + dtype + ".") {}
    };

    class IncompatDevicesForOp : public std::invalid_argument
    {
    public:
        IncompatDevicesForOp(const std::string &op, const std::string lhs_device, const std::string rhs_device) : std::invalid_argument("Cannot run operator " + op + " on incompatible devices " + lhs_device + " and " + rhs_device + ".") {}
    };

    class PybindInvalidArgumentType : public std::invalid_argument
    {
    public:
        PybindInvalidArgumentType(const std::string curr_type, const std::string &expected_type) : std::invalid_argument("Expected an object of type " + expected_type + " but received an object of type " + curr_type + ".") {}
    };

    class CannotUpdateConstArray : public std::runtime_error
    {
    public:
        CannotUpdateConstArray(const std::string &id) : std::runtime_error("Cannot update array " + id + " since it is a constant.") {}
    };
}