#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "cudl/kernels/elementwise.h"
#include "cudl/kernels/gemm.h"
#include "cudl/kernels/reduce.h"
#include "cudl/nn/init.h"
#include "cudl/nn/module.h"

namespace cudl::nn {

enum class Init { HeKaiming, GlorotXavier };

// y = x W + b
// dimensions: x [batch, in], W [in, out], b [out]
// W is [in, out] so forward doesnt transpose and backward transpose
class Linear : public Module {
public:
    Linear(int64_t in_features, int64_t out_features, Rng& rng, Init init = Init::HeKaiming);

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& dy) override;
    std::vector<Parameter*> parameters() override { return {&weight_, &bias_}; }

    Parameter& weight() { return weight_; }
    Parameter& bias() { return bias_; }

private:
    int64_t in_, out_;
    Parameter weight_, bias_; // b = 0 initially
    Tensor x_;   // input of the latest forward, for backward
    Tensor xw_;  // x W, before the bias
    Tensor y_, dx_;
    kernels::GemmFn gemm_;
    kernels::BiasFn add_bias_;
    kernels::ReduceColsFn sum_cols_;
};

class ReLU : public Module {
public:
    ReLU();

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& dy) override;
private:
    Tensor x_;  // input of the latest forward, for backward
    Tensor y_, dx_;
    kernels::UnaryFn relu_;
    kernels::BinaryFn relu_grad_;
};

// Runs its layers in order; backward runs them in reverse.
class Sequential : public Module {
public:
    // Construct a layer in place and return it, e.g. model.add<Linear>(784, 256, rng)
    template <typename M, typename... Args>
    M& add(Args&&... args) {
        auto layer = std::make_unique<M>(std::forward<Args>(args)...);
        M& ref = *layer;
        layers_.push_back(std::move(layer));
        return ref;
    }

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& dy) override;
    std::vector<Parameter*> parameters() override;

    size_t size() const { return layers_.size(); }
    Module& operator[](size_t i) { return *layers_[i]; }
    
private:
    std::vector<std::unique_ptr<Module>> layers_;
};

}  // namespace cudl::nn
