#pragma once

#include "../common.h"

namespace xv::core
{
    struct Id : public IStr
    {
    private:
        usize data;

    public:
        Id() : data(0) {}
        Id(usize id) : data(id) {}
        Id(const Id &id) : data(id.data) {}
        bool operator==(const Id &id) const { return data == id.data; }
        bool operator!=(const Id &id) const { return !(*this == id); }
        Id &operator=(const Id &id)
        {
            data = id.data;
            return *this;
        }
        usize get_data() const { return data; }
        const std::string str() const override { return std::to_string(data); }
    };

    struct IdGenerator
    {
    private:
        static usize counter;

    public:
        IdGenerator() = default;
        IdGenerator(const IdGenerator &) = delete;
        IdGenerator &operator=(const IdGenerator &) = delete;

        Id generate()
        {
            Id curr(counter++);
            return curr;
        }
    };

    inline usize IdGenerator::counter = 1;
}

namespace std
{
    template <>
    struct hash<xv::core::Id>
    {
        std::size_t operator()(const xv::core::Id &id) const
        {
            return std::hash<xv::core::usize>()(id.get_data());
        }
    };
}