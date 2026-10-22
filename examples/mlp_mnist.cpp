// Trains an MLP on MNIST with cuDL's own kernels, then reports test accuracy and the time per step.
// Phase 4 exit check: at least 97% test accuracy.
//
//   scripts/get_mnist.sh
//   build/examples/mlp_mnist [--data data/mnist] [--epochs 5] [--batch 128] [--hidden 512,256]
//                            [--optim adam|sgd] [--lr 1e-3] [--momentum 0.9] [--seed 0] [--target 0.97]
//
// Exits non-zero if test accuracy ends below --target. Every module, the loss and the optimizer
// push NVTX ranges, so `scripts/profile.sh nsys build/examples/mlp_mnist --epochs 1` gives the
// per-op time breakdown of a training step.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "cudl/core/context.h"
#include "cudl/core/nvtx.h"
#include "cudl/core/tensor.h"
#include "cudl/nn/layers.h"
#include "cudl/nn/loss.h"
#include "cudl/optim/optim.h"
#include "mnist.h"

using namespace cudl;

namespace {

struct Args {
    std::string data = "data/mnist";
    int epochs = 5;
    int64_t batch = 128;
    std::vector<int64_t> hidden{512, 256};
    std::string optim = "adam";
    float lr = -1.0f;  // < 0: the optimizer's default below
    float momentum = 0.9f;
    uint64_t seed = 0;
    double target = 0.97;
};

std::vector<int64_t> parse_sizes(const std::string& s) {
    std::vector<int64_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        out.push_back(std::stoll(item));
    }
    return out;
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        CUDL_CHECK(i + 1 < argc, flag << " needs a value");
        const std::string v = argv[++i];
        if (flag == "--data") {
            a.data = v;
        } else if (flag == "--epochs") {
            a.epochs = std::stoi(v);
        } else if (flag == "--batch") {
            a.batch = std::stoll(v);
        } else if (flag == "--hidden") {
            a.hidden = parse_sizes(v);
        } else if (flag == "--optim") {
            a.optim = v;
        } else if (flag == "--lr") {
            a.lr = std::stof(v);
        } else if (flag == "--momentum") {
            a.momentum = std::stof(v);
        } else if (flag == "--seed") {
            a.seed = std::stoull(v);
        } else if (flag == "--target") {
            a.target = std::stod(v);
        } else {
            CUDL_FAIL("unknown flag " << flag);
        }
    }
    CUDL_CHECK(a.optim == "adam" || a.optim == "sgd", "--optim must be adam or sgd");
    CUDL_CHECK(a.batch > 0, "--batch must be positive");
    return a;
}

// Copies samples order[first, first + count) into the batch tensors (host gather, then one upload each)
// TODO: overlap data transfer with another cuda stream
void load_batch(const mnist::Dataset& d, const std::vector<int64_t>& order, int64_t first, int64_t count,
                std::vector<float>& host_x, std::vector<int32_t>& host_y, Tensor& x, Tensor& y) {
    CUDL_NVTX_RANGE("load_batch");
    const int64_t f = mnist::Dataset::kFeatures;
    host_x.resize(static_cast<size_t>(count * f));
    host_y.resize(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        const int64_t s = order[static_cast<size_t>(first + i)];
        const float* src = d.images.data() + s * f;
        std::copy(src, src + f, host_x.begin() + i * f);
        host_y[static_cast<size_t>(i)] = d.labels[static_cast<size_t>(s)];
    }
    if (!x.defined() || x.shape()[0] != count) {
        x = Tensor::empty({count, f});
        y = Tensor::empty({count}, Device::CUDA, DType::I32);
    }
    x.copy_from_host(host_x.data(), host_x.size() * sizeof(float));
    y.copy_from_host(host_y.data(), host_y.size() * sizeof(int32_t));
}

// Fraction of samples whose arg-max logit is the label
double evaluate(nn::Sequential& model, const mnist::Dataset& d, int64_t batch) {
    CUDL_NVTX_RANGE("evaluate");
    std::vector<int64_t> order(static_cast<size_t>(d.n));
    for (int64_t i = 0; i < d.n; ++i) {
        order[static_cast<size_t>(i)] = i;
    }
    std::vector<float> host_x;
    std::vector<int32_t> host_y;
    Tensor x, y;
    int64_t correct = 0;
    for (int64_t first = 0; first < d.n; first += batch) {
        const int64_t count = std::min(batch, d.n - first);
        load_batch(d, order, first, count, host_x, host_y, x, y);
        const std::vector<float> logits = model.forward(x).to_vector();
        for (int64_t i = 0; i < count; ++i) {
            const float* row = logits.data() + i * mnist::Dataset::kClasses;
            const int64_t pred = std::max_element(row, row + mnist::Dataset::kClasses) - row;
            correct += pred == host_y[static_cast<size_t>(i)] ? 1 : 0;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(d.n);
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse(argc, argv);
    const mnist::Dataset train = mnist::load(args.data, "train");
    const mnist::Dataset test = mnist::load(args.data, "t10k");
    std::printf("MNIST: %lld train, %lld test samples\n", static_cast<long long>(train.n),
                static_cast<long long>(test.n));

    nn::Rng rng(args.seed);
    nn::Sequential model;
    std::vector<int64_t> sizes{mnist::Dataset::kFeatures};
    sizes.insert(sizes.end(), args.hidden.begin(), args.hidden.end());
    sizes.push_back(mnist::Dataset::kClasses);
    std::printf("model:");
    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
        model.add<nn::Linear>(sizes[i], sizes[i + 1], rng);
        if (i + 2 < sizes.size()) {
            model.add<nn::ReLU>();
        }
        std::printf(" %lld ->", static_cast<long long>(sizes[i]));
    }
    std::printf(" %lld (ReLU between layers)\n", static_cast<long long>(sizes.back()));

    std::unique_ptr<optim::Optimizer> opt;
    if (args.optim == "adam") {
        opt = std::make_unique<optim::Adam>(model.parameters(), args.lr > 0 ? args.lr : 1e-3f);
    } else {
        opt = std::make_unique<optim::SGD>(model.parameters(), args.lr > 0 ? args.lr : 0.05f, args.momentum);
    }
    std::printf("optimizer: %s, lr %g, batch %lld, %d epochs\n\n", args.optim.c_str(), opt->lr(),
                static_cast<long long>(args.batch), args.epochs);

    nn::CrossEntropyLoss loss_fn;
    std::vector<int64_t> order(static_cast<size_t>(train.n));
    for (int64_t i = 0; i < train.n; ++i) {
        order[static_cast<size_t>(i)] = i;
    }
    std::vector<float> host_x;
    std::vector<int32_t> host_y;
    Tensor x, y;
    const int64_t steps_per_epoch = train.n / args.batch;  // drop the last partial batch
    double total_train_s = 0.0;
    double accuracy = 0.0;

    for (int epoch = 1; epoch <= args.epochs; ++epoch) {
        rng.shuffle(order);
        Context::get().synchronize();
        const auto start = std::chrono::steady_clock::now();
        float last_loss = 0.0f;
        for (int64_t step = 0; step < steps_per_epoch; ++step) {
            CUDL_NVTX_RANGE("train_step");
            load_batch(train, order, step * args.batch, args.batch, host_x, host_y, x, y);
            const Tensor loss = loss_fn.forward(model.forward(x), y);
            model.backward(loss_fn.backward());
            opt->step();
            // Reading the loss synchronizes, so only do it at the end of the epoch
            if (step + 1 == steps_per_epoch) {
                last_loss = loss.to_vector()[0];
            }
        }
        Context::get().synchronize();
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        total_train_s += secs;

        accuracy = evaluate(model, test, 1000);
        std::printf("epoch %d: loss %.4f, test acc %.2f%%, %.2f s (%.3f ms/step)\n", epoch, last_loss,
                    100.0 * accuracy, secs, 1e3 * secs / static_cast<double>(steps_per_epoch));
    }

    const double steps = static_cast<double>(steps_per_epoch) * args.epochs;
    std::printf("\nfinal test accuracy %.2f%% (target %.2f%%), mean %.3f ms/step\n", 100.0 * accuracy,
                100.0 * args.target, 1e3 * total_train_s / steps);
    return accuracy >= args.target ? 0 : 1;
}
