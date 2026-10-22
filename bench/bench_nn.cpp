// Times the training-only kernels: fused softmax cross-entropy over batch x classes shapes, and the
// SGD / Adam updates over parameter sizes. All are memory-bound, so % of peak bandwidth is the number to watch.

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "cudl/core/context.h"
#include "cudl/core/error.h"
#include "cudl/core/tensor.h"
#include "cudl/kernels/loss.h"
#include "cudl/kernels/optim.h"
#include "cudl/kernels/registry.h"
#include "harness.h"

int main(int argc, char** argv) {
    using namespace cudl;
    using namespace cudl::kernels;

    bench::Harness harness(bench::Options::parse(argc, argv));
    const auto& dev = bench::DeviceInfo::get();
    std::printf("# %s, %d SMs, theoretical peak %.0f GB/s\n\n", dev.name.c_str(), dev.sm_count, dev.peak_bw_gbps);
    const cudaStream_t stream = Context::get().stream();

    // (batch, classes): MNIST-sized up to a large vocabulary
    const std::vector<std::pair<int64_t, int64_t>> cross_entropy_shapes{
        {128, 10}, {4096, 10}, {4096, 1000}, {1024, 32768}};
    for (const auto& xs : cross_entropy_shapes) {
        // Plain locals: C++17 lambdas can't capture structured bindings
        const int64_t rows = xs.first;
        const int64_t cols = xs.second;
        Tensor logits = Tensor::zeros({rows, cols});
        Tensor dlogits = Tensor::empty({rows, cols});
        Tensor loss = Tensor::empty({rows});
        Tensor labels = Tensor::zeros({rows}, Device::CUDA, DType::I32);
        const std::string shape = std::to_string(rows) + "x" + std::to_string(cols);
        // Read logits once (re-reads hit cache), write dlogits, read labels, write loss
        const double bytes = 2.0 * static_cast<double>(logits.nbytes()) + 2.0 * static_cast<double>(loss.nbytes());
        const double flops = 4.0 * static_cast<double>(rows) * static_cast<double>(cols);
        for (const auto& k : registry<CrossEntropyLossFn>().entries("cross_entropy_loss")) {
            harness.run("cross_entropy_loss", k, shape, bytes, flops, [&](cudaStream_t s) {
                k.fn(logits.data<float>(), labels.data<int32_t>(), loss.data<float>(), dlogits.data<float>(),
                     static_cast<size_t>(rows), static_cast<size_t>(cols), 1.0f, s);
            });
        }
    }

    const int64_t sizes[] = {int64_t{1} << 18, int64_t{1} << 22, int64_t{1} << 25};
    for (const int64_t n : sizes) {
        Tensor param = Tensor::zeros({n}), grad = Tensor::zeros({n}), m = Tensor::zeros({n}), v = Tensor::zeros({n});
        const size_t un = static_cast<size_t>(n);
        const double nb = static_cast<double>(param.nbytes());
        const std::string shape = std::to_string(n);

        // Read param, grad, velocity; write param, velocity
        for (const auto& k : registry<SgdFn>().entries("sgd")) {
            harness.run("sgd", k, shape, 5 * nb, 3.0 * static_cast<double>(n), [&](cudaStream_t s) {
                k.fn(param.data<float>(), grad.data<float>(), m.data<float>(), un, 0.01f, 0.9f, s);
            });
        }
        // Read param, grad, m, v; write param, m, v
        const AdamStep hp{1e-3f, 0.9f, 0.999f, 1e-8f, 0.1f, 0.001f};
        for (const auto& k : registry<AdamFn>().entries("adam")) {
            harness.run("adam", k, shape, 7 * nb, 12.0 * static_cast<double>(n), [&](cudaStream_t s) {
                k.fn(param.data<float>(), grad.data<float>(), m.data<float>(), v.data<float>(), un, hp, s);
            });
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return 0;
}
