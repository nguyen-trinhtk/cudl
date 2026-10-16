#pragma once

// Benchmark harness: warmup, then per-iteration CUDA-event timing reported as
// median/p10/p90, GB/s, GFLOP/s and % of peak bandwidth. Optionally flushes L2
// between runs and appends JSON lines tagged with GPU name and git commit, so runs
// can be compared over time.

#include <cuda_runtime.h>

#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace cudl::bench {

struct Options {
    int warmup = 10;
    int iters = 100;
    bool flush_l2 = false;  // clear L2 before every timed iteration (cold-cache numbers)
    std::string json_path;  // append one JSON line per result when set (or env CUDL_BENCH_JSON)

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
    std::string op, variant, shape;
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

    // `launch` must enqueue exactly one run of the work on the given stream.
    // Each timed iteration is bracketed by its own CUDA events.
    Result run(const std::string& op, const std::string& variant, const std::string& shape, double bytes, double flops,
               const std::function<void(cudaStream_t)>& launch);

    const std::vector<Result>& results() const { return results_; }

private:
    void report(const Result& r);

    Options opts_;
    void* flush_buf_ = nullptr;
    size_t flush_bytes_ = 0;
    std::ofstream json_;
    std::vector<Result> results_;
    bool header_printed_ = false;
};

}  // namespace cudl::bench
