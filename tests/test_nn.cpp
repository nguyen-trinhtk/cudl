// nn modules and optimizers end to end: Linear and CrossEntropyLoss against the double CPU reference,
// gradcheck of a whole MLP against fp64 central finite differences, and a long-run drift test that
// trains the same model on the GPU (fp32) and the CPU (fp64) and tracks how far the parameters diverge.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "common/compare.h"
#include "common/random.h"
#include "cudl/core/context.h"
#include "cudl/core/tensor.h"
#include "cudl/nn/layers.h"
#include "cudl/nn/loss.h"
#include "cudl/optim/optim.h"
#include "ref/nn_ref.h"
#include "ref/optim_ref.h"

namespace cudl {
namespace {

std::vector<double> to_double(const std::vector<float>& v) {
    return {v.begin(), v.end()};
}

Tensor matrix(const std::vector<float>& data, size_t rows, size_t cols) {
    return Tensor::from_vector(data, {static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
}

std::vector<int32_t> random_labels(size_t n, size_t classes, uint64_t seed) {
    const auto u = test::random_uniform(n, seed, 0.0f, static_cast<float>(classes));
    std::vector<int32_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = std::min(static_cast<int32_t>(u[i]), static_cast<int32_t>(classes) - 1);
    }
    return out;
}

// The model's current weights as reference layers (same values, held in double)
ref::LinearRef ref_of(nn::Linear& l) {
    const Shape& s = l.weight().value.shape();
    ref::LinearRef r;
    r.in = static_cast<size_t>(s[0]);
    r.out = static_cast<size_t>(s[1]);
    r.w = to_double(l.weight().value.to_vector());
    r.b = to_double(l.bias().value.to_vector());
    return r;
}

std::vector<ref::LinearRef> ref_of(nn::Sequential& model) {
    std::vector<ref::LinearRef> out;
    for (size_t i = 0; i < model.size(); ++i) {
        if (auto* l = dynamic_cast<nn::Linear*>(&model[i])) {
            out.push_back(ref_of(*l));
        }
    }
    return out;
}

// Linear, ReLU, ..., Linear
void build_mlp(nn::Sequential& model, const std::vector<int64_t>& sizes, nn::Rng& rng) {
    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
        model.add<nn::Linear>(sizes[i], sizes[i + 1], rng);
        if (i + 2 < sizes.size()) {
            model.add<nn::ReLU>();
        }
    }
}

// fp32 GEMM error grows with the contraction length k (inputs are in [-1, 1]).
test::Tolerance dot_tolerance(size_t k) {
    return {1e-6 * static_cast<double>(k + 1), 1e-5, -1};
}

// ---------------------------------------------------------------------------- Linear

using LinearShape = std::tuple<size_t, size_t, size_t>;  // (batch, in, out)

class LinearTest : public ::testing::TestWithParam<LinearShape> {};

std::string LinearName(const ::testing::TestParamInfo<LinearShape>& info) {
    const auto& [batch, in, out] = info.param;
    return "b" + std::to_string(batch) + "_in" + std::to_string(in) + "_out" + std::to_string(out);
}

TEST_P(LinearTest, ForwardAndBackwardMatchReference) {
    const auto& [batch, in, out] = GetParam();
    nn::Rng rng(batch * 131 + in * 7 + out);
    nn::Linear layer(static_cast<int64_t>(in), static_cast<int64_t>(out), rng);
    // Non-zero bias so the forward check covers it
    const auto bias = test::random_uniform(out, /*seed=*/out);
    layer.bias().value.copy_from_host(bias.data(), bias.size() * sizeof(float));
    const ref::LinearRef r = ref_of(layer);

    const auto host_x = test::random_uniform(batch * in, /*seed=*/batch + in);
    const auto host_dy = test::random_uniform(batch * out, /*seed=*/batch + out + 1);
    const Tensor x = matrix(host_x, batch, in);
    const Tensor dy = matrix(host_dy, batch, out);

    const auto y = layer.forward(x).to_vector();
    const auto dx = layer.backward(dy).to_vector();

    ref::LinearGrads g;
    const auto x_d = to_double(host_x);
    const auto ref_dx = ref::linear_backward(r, x_d, to_double(host_dy), batch, g);
    EXPECT_TRUE(test::AllClose(y, ref::linear_forward(r, x_d, batch), dot_tolerance(in)));
    EXPECT_TRUE(test::AllClose(dx, ref_dx, dot_tolerance(out)));
    EXPECT_TRUE(test::AllClose(layer.weight().grad.to_vector(), g.dw, dot_tolerance(batch)));
    EXPECT_TRUE(test::AllClose(layer.bias().grad.to_vector(), g.db, dot_tolerance(batch)));
}

INSTANTIATE_TEST_SUITE_P(Shapes, LinearTest,
                         ::testing::Values(LinearShape{1, 1, 1}, LinearShape{7, 13, 5}, LinearShape{33, 64, 17},
                                           LinearShape{128, 784, 10}, LinearShape{5, 3, 300}),
                         LinearName);

// ---------------------------------------------------------------------------- CrossEntropyLoss

TEST(CrossEntropyLossTest, MeanLossAndGradMatchReference) {
    const size_t batch = 37, classes = 10;
    const auto logits = test::random_uniform(batch * classes, /*seed=*/1, -5.0f, 5.0f);
    const auto labels = random_labels(batch, classes, /*seed=*/2);

    nn::CrossEntropyLoss loss_fn;
    const Tensor d_labels = Tensor::from_vector(labels, {static_cast<int64_t>(batch)});
    const float loss = loss_fn.forward(matrix(logits, batch, classes), d_labels).to_vector()[0];

    std::vector<double> ref_grad;
    const double ref_loss = ref::mean_cross_entropy(to_double(logits), labels, batch, classes, &ref_grad);
    EXPECT_TRUE(test::AllClose({loss}, {ref_loss}, {1e-5, 1e-5, -1}));
    EXPECT_TRUE(test::AllClose(loss_fn.backward().to_vector(), ref_grad, {1e-6, 1e-5, -1}));
}

// ---------------------------------------------------------------------------- gradcheck

// Every GPU gradient of a small MLP (weights, biases and the input) against fp64 central
// differences of the reference loss. The reference's own analytic gradients are checked the same
// way, since the drift test below relies on them.
TEST(Gradcheck, MlpMatchesFiniteDifferences) {
    const std::vector<int64_t> sizes{6, 5, 4, 3};
    const size_t batch = 4;
    const double h = 1e-6;
    // GPU grads are fp32: compare with a tolerance far above fp32 rounding but far below a wrong formula.
    const test::Tolerance gpu_tol{1e-5, 1e-3, -1};
    const test::Tolerance ref_tol{1e-8, 1e-6, -1};

    nn::Rng rng(42);
    nn::Sequential model;
    build_mlp(model, sizes, rng);
    const auto host_x = test::random_uniform(batch * sizes[0], /*seed=*/7);
    const auto labels = random_labels(batch, static_cast<size_t>(sizes.back()), /*seed=*/8);

    // GPU forward and backward
    nn::CrossEntropyLoss loss_fn;
    const Tensor x = matrix(host_x, batch, static_cast<size_t>(sizes[0]));
    const Tensor logits = model.forward(x);
    const float gpu_loss = loss_fn.forward(logits, Tensor::from_vector(labels, {static_cast<int64_t>(batch)}))
                               .to_vector()[0];
    const auto gpu_dx = model.backward(loss_fn.backward()).to_vector();

    // Reference loss and analytic grads at the same point
    std::vector<ref::LinearRef> layers = ref_of(model);
    std::vector<double> xd = to_double(host_x);
    ref::MlpGrads ref_grads;
    const double ref_loss = ref::mlp_loss(layers, xd, batch, labels, &ref_grads);
    EXPECT_TRUE(test::AllClose({gpu_loss}, {ref_loss}, {1e-6, 1e-5, -1}));

    // Central difference of the reference loss with respect to one value
    auto central_diff = [&](double& v) {
        const double saved = v;
        v = saved + h;
        const double up = ref::mlp_loss(layers, xd, batch, labels);
        v = saved - h;
        const double down = ref::mlp_loss(layers, xd, batch, labels);
        v = saved;
        return (up - down) / (2.0 * h);
    };
    auto numeric_grad = [&](std::vector<double>& values) {
        std::vector<double> out(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            out[i] = central_diff(values[i]);
        }
        return out;
    };

    const std::vector<nn::Parameter*> params = model.parameters();
    ASSERT_EQ(params.size(), 2 * layers.size());
    for (size_t l = 0; l < layers.size(); ++l) {
        const auto num_dw = numeric_grad(layers[l].w);
        const auto num_db = numeric_grad(layers[l].b);
        EXPECT_TRUE(test::AllClose(params[2 * l]->grad.to_vector(), num_dw, gpu_tol)) << "layer " << l << " dW";
        EXPECT_TRUE(test::AllClose(params[2 * l + 1]->grad.to_vector(), num_db, gpu_tol)) << "layer " << l << " db";
        // AllClose takes float actuals; the reference grads are well within float range.
        const std::vector<float> ref_dw(ref_grads.layers[l].dw.begin(), ref_grads.layers[l].dw.end());
        const std::vector<float> ref_db(ref_grads.layers[l].db.begin(), ref_grads.layers[l].db.end());
        EXPECT_TRUE(test::AllClose(ref_dw, num_dw, ref_tol)) << "reference layer " << l << " dW";
        EXPECT_TRUE(test::AllClose(ref_db, num_db, ref_tol)) << "reference layer " << l << " db";
    }
    EXPECT_TRUE(test::AllClose(gpu_dx, numeric_grad(xd), gpu_tol)) << "input grad";
}

// ---------------------------------------------------------------------------- training

// Adam on one fixed batch: the model must be able to memorize it.
TEST(Training, AdamOverfitsOneBatch) {
    const std::vector<int64_t> sizes{20, 32, 5};
    const size_t batch = 16;
    nn::Rng rng(3);
    nn::Sequential model;
    build_mlp(model, sizes, rng);
    optim::Adam opt(model.parameters(), /*lr=*/1e-2f);
    nn::CrossEntropyLoss loss_fn;

    const Tensor x = matrix(test::random_uniform(batch * sizes[0], /*seed=*/4), batch, static_cast<size_t>(sizes[0]));
    const Tensor labels = Tensor::from_vector(random_labels(batch, static_cast<size_t>(sizes.back()), 5),
                                              {static_cast<int64_t>(batch)});
    float first = 0.0f, last = 0.0f;
    for (int step = 0; step < 200; ++step) {
        const Tensor loss = loss_fn.forward(model.forward(x), labels);
        model.backward(loss_fn.backward());
        opt.step();
        if (step == 0) {
            first = loss.to_vector()[0];
        }
        last = loss.to_vector()[0];
    }
    EXPECT_LT(last, 0.05f * first) << "first loss " << first << ", last " << last;
}

// Trains the same MLP with SGD + momentum on the GPU (fp32) and the CPU reference (fp64), from the same
// weights on the same batches, and tracks the parameter divergence max|gpu - cpu| / max|cpu| per step.
// fp32 rounding makes the two drift apart; the test checks that the drift stays bounded.
// Set CUDL_TRAIN_DRIFT_CSV=path to save the per-step table.
TEST(TrainingDrift, GpuTracksFp64Reference) {
    const std::vector<int64_t> sizes{32, 64, 64, 10};
    const size_t batch = 32, num_batches = 8;
    const int steps = 200;
    const float lr = 0.05f, momentum = 0.9f;
    const double max_divergence = 1e-3;

    nn::Rng rng(11);
    nn::Sequential model;
    build_mlp(model, sizes, rng);
    optim::SGD opt(model.parameters(), lr, momentum);
    nn::CrossEntropyLoss loss_fn;

    std::vector<ref::LinearRef> layers = ref_of(model);
    std::vector<std::vector<double>> vel_w, vel_b;
    for (const auto& l : layers) {
        vel_w.emplace_back(l.w.size(), 0.0);
        vel_b.emplace_back(l.b.size(), 0.0);
    }

    const size_t in = static_cast<size_t>(sizes[0]);
    const size_t classes = static_cast<size_t>(sizes.back());
    std::vector<std::vector<float>> xs;
    std::vector<std::vector<int32_t>> ys;
    for (size_t i = 0; i < num_batches; ++i) {
        xs.push_back(test::random_uniform(batch * in, /*seed=*/100 + i));
        ys.push_back(random_labels(batch, classes, /*seed=*/200 + i));
    }
    Tensor x = Tensor::empty({static_cast<int64_t>(batch), sizes[0]});
    Tensor labels = Tensor::empty({static_cast<int64_t>(batch)}, Device::CUDA, DType::I32);

    const std::vector<nn::Parameter*> params = model.parameters();
    std::string csv = "step,gpu_loss,cpu_loss,divergence\n";
    std::printf("%6s %12s %12s %12s\n", "step", "gpu_loss", "cpu_loss", "divergence");
    double divergence = 0.0;
    for (int step = 1; step <= steps; ++step) {
        const size_t b = static_cast<size_t>(step) % num_batches;
        x.copy_from_host(xs[b].data(), xs[b].size() * sizeof(float));
        labels.copy_from_host(ys[b].data(), ys[b].size() * sizeof(int32_t));
        const Tensor loss = loss_fn.forward(model.forward(x), labels);
        model.backward(loss_fn.backward());
        opt.step();

        ref::MlpGrads g;
        const double cpu_loss = ref::mlp_loss(layers, to_double(xs[b]), batch, ys[b], &g);
        for (size_t l = 0; l < layers.size(); ++l) {
            ref::sgd_step(layers[l].w, g.layers[l].dw, vel_w[l], lr, momentum);
            ref::sgd_step(layers[l].b, g.layers[l].db, vel_b[l], lr, momentum);
        }

        double max_diff = 0.0, max_ref = 0.0;
        for (size_t l = 0; l < layers.size(); ++l) {
            const auto w = params[2 * l]->value.to_vector();
            const auto bias = params[2 * l + 1]->value.to_vector();
            for (size_t i = 0; i < w.size(); ++i) {
                max_diff = std::max(max_diff, std::fabs(w[i] - layers[l].w[i]));
                max_ref = std::max(max_ref, std::fabs(layers[l].w[i]));
            }
            for (size_t i = 0; i < bias.size(); ++i) {
                max_diff = std::max(max_diff, std::fabs(bias[i] - layers[l].b[i]));
                max_ref = std::max(max_ref, std::fabs(layers[l].b[i]));
            }
        }
        divergence = max_diff / max_ref;
        const float gpu_loss = loss.to_vector()[0];

        char line[128];
        std::snprintf(line, sizeof line, "%d,%.9g,%.9g,%.3e\n", step, gpu_loss, cpu_loss, divergence);
        csv += line;
        if (step % 20 == 0 || step == 1) {
            std::printf("%6d %12.6f %12.6f %12.3e\n", step, gpu_loss, cpu_loss, divergence);
        }
        EXPECT_LT(divergence, max_divergence) << "at step " << step;
        if (divergence >= max_divergence) break;
    }

    if (const char* path = std::getenv("CUDL_TRAIN_DRIFT_CSV")) {
        std::ofstream(path) << csv;
        std::printf("wrote %s\n", path);
    }
}

}  // namespace
}  // namespace cudl
