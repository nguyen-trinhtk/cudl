#pragma once

#include <cstdint>
#include <vector>

#include "cudl/core/tensor.h"
#include "cudl/kernels/optim.h"
#include "cudl/nn/module.h"

namespace cudl::optim {

// Updates parameters in place from the grads the latest backward() wrote.
// The parameters must outlive the optimizer.
class Optimizer {
public:
    explicit Optimizer(std::vector<nn::Parameter*> params);
    virtual ~Optimizer() = default;

    virtual void step() = 0;

    float lr() const { return lr_; }
    void set_lr(float lr) { lr_ = lr; }

protected:
    std::vector<nn::Parameter*> params_;
    float lr_ = 0.0f;
};

// SGD with optional momentum (see kernels::SgdFn for the update rule)
class SGD final : public Optimizer {
public:
    SGD(std::vector<nn::Parameter*> params, float lr, float momentum = 0.0f);
    void step() override;

private:
    float momentum_;
    std::vector<Tensor> velocity_;  // one per parameter, zero-initialized; empty without momentum
    kernels::SgdFn sgd_;
};

// Adam (see kernels::AdamFn for the update rule)
class Adam final : public Optimizer {
public:
    Adam(std::vector<nn::Parameter*> params, float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
         float eps = 1e-8f);
    void step() override;

private:
    float beta1_, beta2_, eps_;
    int64_t t_ = 0;  // steps taken
    std::vector<Tensor> m_, v_;  // first and second moments, one per parameter
    kernels::AdamFn adam_;
};

}  // namespace cudl::optim
