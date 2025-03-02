#include <exception>
#include <string>

namespace xv::core
{
    class MTLGraphNotCompiledException : public std::exception
    {
    private:
        std::string msg;

    public:
        MTLGraphNotCompiledException() : msg("Graph has not been compiled.") {}
        const char *what() const noexcept override { return msg.c_str(); }
    };
}