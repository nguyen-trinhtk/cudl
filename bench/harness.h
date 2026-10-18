#pragma once

// Benchmark harness: warmup, then per-iteration CUDA-event timing reported as
// median/p10/p90, GB/s, GFLOP/s and % of peak bandwidth. Optionally flushes L2
// between runs and appends JSON lines tagged with GPU name and git commit, so runs
// can be compared over time.

#include <cuda_runtime.h>

#include <fstream>
#include <functional>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include "cudl/kernels/registry.h"

namespace cudl::bench {

struct Options {
    int warmup = 10;
    int iters = 100;
    bool flush_l2 = false;  // clear L2 before every timed iteration (cold-cache numbers)
    std::string json_path;  // append one JSON line per result when set (or env CUDL_BENCH_JSON)

    // Axis filters: regexes that must match the whole op / algo / load name (empty = all)
    std::string op_filter, algo_filter, load_filter;

    static Options parse(int argc, char** argv);
};

struct DeviceInfo {
    std::string name;
    int sm_count = 0;
    size_t l2_bytes = 0;
    double peak_bw_gbps = 0.0;  // theoretical; override with env CUDL_PEAK_BW_GBPS

    static const DeviceInfo& get();
};

struct Result {
    std::string op, algo, load, variant, shape;
    double bytes = 0.0;  // bytes moved per launch, from the op's cost model
    double flops = 0.0;
    double median_ms = 0.0, p10_ms = 0.0, p90_ms = 0.0;

    double gbps() const { return median_ms > 0 ? bytes / (median_ms * 1e6) : 0.0; }
    double gflops() const { return median_ms > 0 ? flops / (median_ms * 1e6) : 0.0; }
    double pct_peak_bw() const;
};

class Harness {
public:
    explicit Harness(Options opts);
    ~Harness();
    Harness(const Harness&) = delete;
    Harness& operator=(const Harness&) = delete;

    using Launch = std::function<void(cudaStream_t)>;

    // Time one registered kernel; `op` is the reported name (e.g. "sum_rows" for k.op "sum").
    // Returns nullopt, without launching, if the --op/--algo/--load filters exclude it.
    // `launch` must enqueue exactly one run of the work on the given stream.
    template <typename Fn>
    std::optional<Result> run(const std::string& op, const kernels::KernelEntry<Fn>& k, const std::string& shape,
                              double bytes, double flops, const Launch& launch) {
        if (!selected(op, k.algo, k.load)) return std::nullopt;
        return time({op, k.algo, k.load, k.variant, shape, bytes, flops}, launch);
    }

    // Time a reference implementation (e.g. cudaMemcpy). Always runs: filters only select kernels.
    Result run_baseline(const std::string& op, const std::string& name, const std::string& shape,
                        double bytes, double flops, const Launch& launch);

    // True if the axis filters keep this kernel
    bool selected(const std::string& op, const std::string& algo, const std::string& load) const;

    const std::vector<Result>& results() const { return results_; }

private:
    // Each timed iteration is bracketed by its own CUDA events.
    Result time(Result r, const Launch& launch);
    void report(const Result& r);

    Options opts_;
    std::regex op_re_, algo_re_, load_re_;
    void* flush_buf_ = nullptr;
    size_t flush_bytes_ = 0;
    std::ofstream json_;
    std::vector<Result> results_;
    bool header_printed_ = false;
};

}  // namespace cudl::bench
