#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "cudl/core/error.h"

// Kernel registry
namespace cudl::kernels {

// Load axis names (match enum class Load in src/kernels/launch_utils.h)
inline constexpr const char* kLoadScalar = "scalar";
inline constexpr const char* kLoadVec4 = "vec4";

// One implementation of an op, identified by three independent axes:
//   op:   what is computed ("sum", "relu", ...)
//   algo: how it is computed ("atomic", "smem", "warp", ...); empty if the op has no algorithm axis
//   load: how each thread reads memory (kLoadScalar or kLoadVec4)
// Strings rather than enums so one registry type serves every op and benchmarks can filter on them.
template <typename Fn>
struct KernelEntry {
    std::string op;
    std::string algo;
    std::string load = kLoadScalar;
    Fn fn = nullptr;            // launcher as a function obj
    bool deterministic = true;  // Note: should be false for kernels whose results depend on atomics ordering
    std::string variant{};      // unique name within the op, set by Registry::add (see variant_name)
};

// Variant name from the algo and load axes:
//   algo, scalar -> "algo"     algo, vec4 -> "algo_vec4"
//   no algo      -> the load name ("scalar" / "vec4")
inline std::string variant_name(const std::string& algo, const std::string& load) {
    if (algo.empty()) return load;
    return load == kLoadScalar ? algo : algo + "_" + load;
}

template <typename Fn>
class Registry {
public:
    void add(KernelEntry<Fn> entry) {
        CUDL_CHECK(!entry.load.empty(), "kernel " << entry.op << " has no load axis");
        entry.variant = variant_name(entry.algo, entry.load);
        auto& list = by_op_[entry.op];
        for (const auto& e : list)
            CUDL_CHECK(e.variant != entry.variant, "duplicate kernel " << entry.op << "/" << entry.variant);
        list.push_back(std::move(entry));
    }

    const KernelEntry<Fn>& get(const std::string& op, const std::string& variant) const {
        for (const auto& e : entries(op))
            if (e.variant == variant) return e;
        CUDL_FAIL("unknown variant '" << variant << "' for op '" << op << "'");
    }

    // First registered variant
    // can be overridden by env CUDL_KERNEL_<OP>=<variant>.
    const KernelEntry<Fn>& get_default(const std::string& op) const {
        std::string env = "CUDL_KERNEL_" + op;
        std::transform(env.begin(), env.end(), env.begin(), [](unsigned char c) { return std::toupper(c); });
        if (const char* v = std::getenv(env.c_str())) return get(op, v);
        return entries(op).front();
    }

    // First variant (if available)
    const std::vector<KernelEntry<Fn>>& entries(const std::string& op) const {
        auto it = by_op_.find(op);
        CUDL_CHECK(it != by_op_.end() && !it->second.empty(), "no kernels registered for op '" << op << "'");
        return it->second;
    }

    // All variants of one op
    std::vector<std::string> variants(const std::string& op) const {
        std::vector<std::string> out;
        for (const auto& e : entries(op)) out.push_back(e.variant);
        return out;
    }

private:
    // Hash map
    std::map<std::string, std::vector<KernelEntry<Fn>>> by_op_;
};

// Namespaced utils for creating registry & register kernels
namespace detail {
// Used by register_*_kernels(); 
// Note: everything else uses registry<Fn>().
template <typename Fn>
Registry<Fn>& raw_registry() {
    static Registry<Fn> r;
    return r;
}

void ensure_kernels_registered(); // see .cpp
}  // namespace detail

// Global singleton registry
// Template: diff args for diff ops
template <typename Fn>
Registry<Fn>& registry() {
    detail::ensure_kernels_registered(); // register all kernels once
    return detail::raw_registry<Fn>(); // return 
}
}  // namespace cudl::kernels
