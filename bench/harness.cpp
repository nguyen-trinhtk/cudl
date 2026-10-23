#include "harness.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "cudl/core/context.h"
#include "cudl/core/error.h"
#include "cudl/core/nvtx.h"

#ifndef CUDL_GIT_SHA
#define CUDL_GIT_SHA "unknown"
#endif

namespace cudl::bench {
namespace {

[[noreturn]] void usage(const char* prog, int code) {
    std::fprintf(code ? stderr : stdout,
                 "usage: %s [--warmup N] [--iters N] [--flush-l2] [--json PATH]\n"
                 "          [--op REGEX] [--algo REGEX] [--load REGEX] [--size N]...\n"
                 "  --op/--algo/--load keep only kernels whose op / algo / load name fully matches\n"
                 "  (load is scalar or vec4; algo is empty for ops without one, e.g. elementwise)\n"
                 "  --size replaces the bench's default problem sizes (repeatable; gemm only for now)\n",
                 prog);
    std::exit(code);
}

// Nearest-rank percentile of sorted samples.
double percentile(const std::vector<float>& sorted, double p) {
    return sorted[static_cast<size_t>(p * (sorted.size() - 1) + 0.5)];
}

std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}

std::string git_sha() {
    if (const char* env = std::getenv("CUDL_GIT_SHA")) return env;
    return CUDL_GIT_SHA;
}

std::string utc_timestamp() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return buf;
}

}  // namespace

Options Options::parse(int argc, char** argv) {
    Options o;
    if (const char* env = std::getenv("CUDL_BENCH_JSON")) o.json_path = env;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> const char* {
            if (i + 1 >= argc) usage(argv[0], 2);
            return argv[++i];
        };
        if (arg == "--warmup")
            o.warmup = std::atoi(value());
        else if (arg == "--iters")
            o.iters = std::atoi(value());
        else if (arg == "--flush-l2")
            o.flush_l2 = true;
        else if (arg == "--json")
            o.json_path = value();
        else if (arg == "--op")
            o.op_filter = value();
        else if (arg == "--algo")
            o.algo_filter = value();
        else if (arg == "--load")
            o.load_filter = value();
        else if (arg == "--size")
            o.sizes.push_back(std::atoll(value()));
        else if (arg == "-h" || arg == "--help")
            usage(argv[0], 0);
        else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage(argv[0], 2);
        }
    }
    if (o.iters < 1 || o.warmup < 0) usage(argv[0], 2);
    return o;
}

const DeviceInfo& DeviceInfo::get() {
    static const DeviceInfo info = [] {
        DeviceInfo d;
        const int dev = Context::get().device();
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        d.name = prop.name;
        d.sm_count = prop.multiProcessorCount;
        d.l2_bytes = static_cast<size_t>(prop.l2CacheSize);

        int mem_clock_khz = 0, bus_width_bits = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
        d.peak_bw_gbps = 2.0 * mem_clock_khz * 1e3 * (bus_width_bits / 8.0) / 1e9;  // double data rate
        if (const char* env = std::getenv("CUDL_PEAK_BW_GBPS")) d.peak_bw_gbps = std::atof(env);
        return d;
    }();
    return info;
}

double Result::pct_peak_bw() const {
    const double peak = DeviceInfo::get().peak_bw_gbps;
    return peak > 0 ? 100.0 * gbps() / peak : 0.0;
}

Harness::Harness(Options opts) : opts_(std::move(opts)) {
    try {
        op_re_ = std::regex(opts_.op_filter.empty() ? ".*" : opts_.op_filter);
        algo_re_ = std::regex(opts_.algo_filter.empty() ? ".*" : opts_.algo_filter);
        load_re_ = std::regex(opts_.load_filter.empty() ? ".*" : opts_.load_filter);
    } catch (const std::regex_error& e) {
        CUDL_FAIL("bad --op/--algo/--load regex: " << e.what());
    }
    if (opts_.flush_l2) {
        flush_bytes_ = std::max<size_t>(2 * DeviceInfo::get().l2_bytes, size_t{1} << 20);
        CUDA_CHECK(cudaMalloc(&flush_buf_, flush_bytes_));
    }
    if (!opts_.json_path.empty()) {
        json_.open(opts_.json_path, std::ios::app);
        CUDL_CHECK(json_.is_open(), "cannot open " << opts_.json_path);
    }
}

Harness::~Harness() {
    if (flush_buf_) cudaFree(flush_buf_);
}

bool Harness::selected(const std::string& op, const std::string& algo, const std::string& load) const {
    return std::regex_match(op, op_re_) && std::regex_match(algo, algo_re_) && std::regex_match(load, load_re_);
}

Result Harness::run_baseline(const std::string& op, const std::string& name, const std::string& shape, double bytes,
                             double flops, const Launch& launch) {
    return time({op, name, "-", name, shape, bytes, flops}, launch);
}

Result Harness::time(Result r, const Launch& launch) {
    CUDL_NVTX_RANGE((r.op + "/" + r.variant + "/" + r.shape).c_str());
    const cudaStream_t stream = Context::get().stream();

    for (int i = 0; i < opts_.warmup; ++i) launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<cudaEvent_t> start(opts_.iters), stop(opts_.iters);
    for (int i = 0; i < opts_.iters; ++i) {
        CUDA_CHECK(cudaEventCreate(&start[i]));
        CUDA_CHECK(cudaEventCreate(&stop[i]));
    }
    for (int i = 0; i < opts_.iters; ++i) {
        if (flush_buf_) CUDA_CHECK(cudaMemsetAsync(flush_buf_, i & 0xff, flush_bytes_, stream));
        CUDA_CHECK(cudaEventRecord(start[i], stream));
        launch(stream);
        CUDA_CHECK(cudaEventRecord(stop[i], stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> times(opts_.iters);
    for (int i = 0; i < opts_.iters; ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start[i], stop[i]));
        cudaEventDestroy(start[i]);
        cudaEventDestroy(stop[i]);
    }
    std::sort(times.begin(), times.end());

    r.median_ms = percentile(times, 0.5);
    r.p10_ms = percentile(times, 0.1);
    r.p90_ms = percentile(times, 0.9);
    results_.push_back(r);
    report(r);
    return r;
}

void Harness::report(const Result& r) {
    if (!header_printed_) {
        std::printf("%-10s %-10s %-6s %-14s %10s %10s %10s %9s %9s %8s\n", "op", "algo", "load", "shape", "median_ms",
                    "p10_ms", "p90_ms", "GB/s", "GFLOP/s", "%peakBW");
        header_printed_ = true;
    }
    std::printf("%-10s %-10s %-6s %-14s %10.4f %10.4f %10.4f %9.1f %9.1f %7.1f%%\n", r.op.c_str(),
                r.algo.empty() ? "-" : r.algo.c_str(), r.load.c_str(), r.shape.c_str(), r.median_ms, r.p10_ms,
                r.p90_ms, r.gbps(), r.gflops(), r.pct_peak_bw());
    std::fflush(stdout);

    if (json_.is_open()) {
        const DeviceInfo& dev = DeviceInfo::get();
        json_ << "{\"op\":\"" << json_escape(r.op) << "\",\"variant\":\"" << json_escape(r.variant) << "\",\"algo\":\""
              << json_escape(r.algo) << "\",\"load\":\"" << json_escape(r.load) << "\",\"shape\":\""
              << json_escape(r.shape) << "\",\"median_ms\":" << r.median_ms << ",\"p10_ms\":" << r.p10_ms
              << ",\"p90_ms\":" << r.p90_ms << ",\"bytes\":" << r.bytes << ",\"flops\":" << r.flops
              << ",\"gbps\":" << r.gbps() << ",\"gflops\":" << r.gflops() << ",\"pct_peak_bw\":" << r.pct_peak_bw()
              << ",\"iters\":" << opts_.iters << ",\"flush_l2\":" << (opts_.flush_l2 ? "true" : "false")
              << ",\"gpu\":\"" << json_escape(dev.name) << "\",\"git_sha\":\"" << json_escape(git_sha())
              << "\",\"timestamp\":\"" << utc_timestamp() << "\"}\n";
        json_.flush();
    }
}

}  // namespace cudl::bench
