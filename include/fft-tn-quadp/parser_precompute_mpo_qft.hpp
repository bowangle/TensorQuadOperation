#pragma once
// bench_args.hpp — parser for the "+flag key=value" grammar.
//
// Grammar:
//   +double | +float128 | +dd128     enable a type (all default to false)
//   +all                             enable all three types
//   +compress | +C                   enable compression (default false)
//   min_a=<int>                      default 10
//   max_a=<int>                      default 100
//   chi=<int>                        default 35
//   eps_factor=<int> | ef=<int>      default 2 (tolerance = 10^N * epsilon)
//   -h | --help | help               print usage and exit(0)
//
// Example:
//   ./bench +float128 +dd128 +C min_a=20 max_a=500 chi=50
//
// Requires C++17.

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

struct BenchArgs {
    bool use_double   = false;
    bool use_float128 = false;
    bool use_dd128    = false;
    bool compress     = false;
    int  min_a        = 10;
    int  max_a        = 100;
    int  chi          = 35;
    int  eps_factor   = 2;   // tolerance = 10^eps_factor * epsilon of the type

    bool any_type() const { return use_double || use_float128 || use_dd128; }
};

namespace detail {

inline int parse_int(std::string_view key, std::string_view value) {
    if (value.empty()) {
        throw std::runtime_error(std::string(key) + " needs a value, e.g. " +
                                 std::string(key) + "=42");
    }
    int result = 0;
    size_t consumed = 0;
    try {
        result = std::stoi(std::string(value), &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid integer for " + std::string(key) +
                                 ": '" + std::string(value) + "'");
    }
    if (consumed != value.size()) {
        throw std::runtime_error("trailing characters in " + std::string(key) +
                                 ": '" + std::string(value) + "'");
    }
    return result;
}

}  // namespace detail

inline void print_bench_usage(const char* prog) {
    std::cout <<
        "Usage: " << prog << " [+double] [+float128] [+dd128] [+all]"
        " [+compress|+C] [min_a=N] [max_a=N] [chi=N] [eps_factor=N]\n"
        "\n"
        "Type toggles (all off by default):\n"
        "  +double       enable double\n"
        "  +float128     enable float128\n"
        "  +dd128        enable dd128 (double-double)\n"
        "  +all          enable all of the above (types only)\n"
        "\n"
        "Other toggles:\n"
        "  +compress, +C enable compression (off by default)\n"
        "\n"
        "Parameters (defaults: min_a=10, max_a=100, chi=35, eps_factor=2):\n"
        "  min_a=N       lower bound\n"
        "  max_a=N       upper bound (must be >= min_a)\n"
        "  chi=N         chi value\n"
        "  eps_factor=N  tolerance exponent: tolerance = 10^N * epsilon\n"
        "  ef=N          short form of eps_factor\n";
}

// Parses argv into a BenchArgs. Throws std::runtime_error on bad input.
// Prints usage and exits(0) on -h/--help/help.
inline BenchArgs parse_bench_args(int argc, char* argv[]) {
    BenchArgs args;

    for (int i = 1; i < argc; ++i) {
        std::string_view tok = argv[i];

        if (tok == "-h" || tok == "--help" || tok == "help") {
            print_bench_usage(argv[0]);
            std::exit(0);
        }

        if (!tok.empty() && tok.front() == '+') {
            std::string_view name = tok.substr(1);
            if      (name == "double")   args.use_double   = true;
            else if (name == "float128") args.use_float128 = true;
            else if (name == "dd128")    args.use_dd128    = true;
            else if (name == "compress" || name == "C") {
                args.compress = true;
            }
            else if (name == "all") {
                args.use_double = args.use_float128 = args.use_dd128 = true;
            } else {
                throw std::runtime_error("unknown type flag: +" +
                                         std::string(name));
            }
            continue;
        }

        if (auto eq = tok.find('='); eq != std::string_view::npos) {
            std::string_view key   = tok.substr(0, eq);
            std::string_view value = tok.substr(eq + 1);
            if      (key == "min_a") args.min_a = detail::parse_int(key, value);
            else if (key == "max_a") args.max_a = detail::parse_int(key, value);
            else if (key == "chi")   args.chi   = detail::parse_int(key, value);
            else if (key == "eps_factor" || key == "ef") {
                args.eps_factor = detail::parse_int(key, value);
            }
            else {
                throw std::runtime_error("unknown parameter: " +
                                         std::string(key));
            }
            continue;
        }

        throw std::runtime_error("unrecognized argument: '" + std::string(tok) +
                                 "' (expected +flag or key=value)");
    }

    if (args.min_a > args.max_a) {
        throw std::runtime_error("min_a (" + std::to_string(args.min_a) +
                                 ") must be <= max_a (" +
                                 std::to_string(args.max_a) + ")");
    }
    return args;
}