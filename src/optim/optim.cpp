#include "cudl/optim/optim.h"

#include <cmath>
#include <utility>

#include "cudl/core/context.h"
#include "cudl/core/nvtx.h"
#include "cudl/kernels/registry.h"

namespace cudl::optim {

Optimizer::Optimizer(std::vector<nn::Parameter*> params) : params_(std::move(params)) {
    for (const nn::Parameter* p : params_) {
        CUDL_CHECK(p != nullptr, "null parameter");
        CUDL_CHECK(p->value.shape() == p->grad.shape(), "parameter '" << p->name << "' grad shape mismatch");
    }
}

// ---- SGD ----

SGD::SGD(std::vector<nn::Parameter*> params, float lr, float momentum)
    : Optimizer(std::move(params)), momentum_(momentum) {
    lr_ = lr;
    if (momentum_ != 0.0f) {
        for (const nn::Parameter* p : params_) {
            velocity_.push_back(Tensor::zeros(p->value.shape()));
        }
    }
    sgd_ = kernels::registry<kernels::SgdFn>().get_default("sgd").fn;
}

void SGD::step() {
    CUDL_NVTX_RANGE("sgd.step");
    const cudaStream_t stream = Context::get().stream();
    for (size_t i = 0; i < params_.size(); ++i) {
        nn::Parameter& p = *params_[i];
        float* velocity = velocity_.empty() ? nullptr : velocity_[i].data<float>();
        const size_t n = static_cast<size_t>(p.value.numel());
        sgd_(p.value.data<float>(), p.grad.data<float>(), velocity, n, lr_, momentum_, stream);
    }
}

// ---- Adam ----

Adam::Adam(std::vector<nn::Parameter*> params, float lr, float beta1, float beta2, float eps)
    : Optimizer(std::move(params)), beta1_(beta1), beta2_(beta2), eps_(eps) {
    lr_ = lr;
    for (const nn::Parameter* p : params_) {
        m_.push_back(Tensor::zeros(p->value.shape()));
        v_.push_back(Tensor::zeros(p->value.shape()));
    }
    adam_ = kernels::registry<kernels::AdamFn>().get_default("adam").fn;
}

void Adam::step() {
    CUDL_NVTX_RANGE("adam.step");
    ++t_;
    const double t = static_cast<double>(t_);
    const double bc1 = 1.0 - std::pow(static_cast<double>(beta1_), t);
    const double bc2 = 1.0 - std::pow(static_cast<double>(beta2_), t);
    const kernels::AdamStep hp{lr_, beta1_, beta2_, eps_, static_cast<float>(bc1), static_cast<float>(bc2)};

    const cudaStream_t stream = Context::get().stream();
    for (size_t i = 0; i < params_.size(); ++i) {
        nn::Parameter& p = *params_[i];
        const size_t n = static_cast<size_t>(p.value.numel());
        adam_(p.value.data<float>(), p.grad.data<float>(), m_[i].data<float>(), v_[i].data<float>(), n, hp, stream);
    }
}

}  // namespace cudl::optim
