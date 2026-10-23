#include "cudl/nn/layers.h"

#include "buffers.h"
#include "cudl/core/context.h"
#include "cudl/core/nvtx.h"
#include "cudl/kernels/registry.h"

namespace cudl::nn {

using detail::check_matrix;
using detail::ensure;

// ---- Linear ----

Linear::Linear(int64_t in_features, int64_t out_features, Rng& rng, Init init)
    : in_(in_features), out_(out_features) {
    CUDL_CHECK(in_ > 0 && out_ > 0, "Linear(" << in_ << ", " << out_ << "): sizes must be positive");
    const size_t fan_in = static_cast<size_t>(in_);
    const size_t fan_out = static_cast<size_t>(out_);
    const size_t n = fan_in * fan_out;
    const std::vector<float> w =
        init == Init::HeKaiming ? he_kaiming_uniform(n, fan_in, rng) : glorot_xavier_uniform(n, fan_in, fan_out, rng);

    weight_ = {"weight", Tensor::from_vector(w, {in_, out_}), Tensor::zeros({in_, out_})};
    bias_ = {"bias", Tensor::zeros({out_}), Tensor::zeros({out_})};

    gemm_ = kernels::registry<kernels::GemmFn>().get_default("gemm").fn;
    add_bias_ = kernels::registry<kernels::BiasFn>().get_default("add_bias", kernels::kLoadVec4).fn;
    sum_cols_ = kernels::registry<kernels::ReduceColsFn, kernels::ReduceColsTag>().get_default("sum", "thread").fn;
}

Tensor Linear::forward(const Tensor& x) {
    CUDL_NVTX_RANGE("linear.forward");
    check_matrix(x, in_, "Linear input");
    const int64_t batch = x.shape()[0];
    x_ = x;
    ensure(xw_, {batch, out_});
    ensure(y_, {batch, out_});

    const cudaStream_t stream = Context::get().stream();
    const size_t b = static_cast<size_t>(batch);
    const size_t in = static_cast<size_t>(in_);
    const size_t out = static_cast<size_t>(out_);
    // xw = x W, then y = xw + b (a GEMM epilogue will fuse these in Phase 5)
    gemm_(x.data<float>(), weight_.value.data<float>(), xw_.data<float>(), b, out, in, false, false, 1.0f, 0.0f,
          stream);
    add_bias_(xw_.data<float>(), bias_.value.data<float>(), y_.data<float>(), b, out, stream);
    return y_;
}

Tensor Linear::backward(const Tensor& dy) {
    CUDL_NVTX_RANGE("linear.backward");
    CUDL_CHECK(x_.defined(), "Linear::backward called before forward");
    check_matrix(dy, out_, "Linear output grad");
    const int64_t batch = x_.shape()[0];
    CUDL_CHECK(dy.shape()[0] == batch, "output grad batch " << dy.shape()[0] << " != input batch " << batch);
    ensure(dx_, {batch, in_});

    const cudaStream_t stream = Context::get().stream();
    const size_t b = static_cast<size_t>(batch);
    const size_t in = static_cast<size_t>(in_);
    const size_t out = static_cast<size_t>(out_);
    // dW = x^T dy  [in, out]
    gemm_(x_.data<float>(), dy.data<float>(), weight_.grad.data<float>(), in, out, b, true, false, 1.0f, 0.0f,
          stream);
    // db = dy summed over the batch  [out]
    sum_cols_(dy.data<float>(), bias_.grad.data<float>(), b, out, stream);
    // dx = dy W^T  [batch, in]
    gemm_(dy.data<float>(), weight_.value.data<float>(), dx_.data<float>(), b, in, out, false, true, 1.0f, 0.0f,
          stream);
    return dx_;
}

// ---- ReLU ----

ReLU::ReLU() {
    relu_ = kernels::registry<kernels::UnaryFn>().get_default("relu", kernels::kLoadVec4).fn;
    relu_grad_ = kernels::registry<kernels::BinaryFn>().get_default("relu_grad", kernels::kLoadVec4).fn;
}

Tensor ReLU::forward(const Tensor& x) {
    CUDL_NVTX_RANGE("relu.forward");
    CUDL_CHECK(x.defined() && x.dtype() == DType::F32 && x.is_contiguous(), "ReLU input must be contiguous f32");
    x_ = x;
    ensure(y_, x.shape());
    const size_t n = static_cast<size_t>(x.numel());
    relu_(x.data<float>(), y_.data<float>(), n, Context::get().stream());
    return y_;
}

Tensor ReLU::backward(const Tensor& dy) {
    CUDL_NVTX_RANGE("relu.backward");
    CUDL_CHECK(x_.defined(), "ReLU::backward called before forward");
    CUDL_CHECK(dy.shape() == x_.shape(), "ReLU output grad shape must match the input shape");
    ensure(dx_, x_.shape());
    const size_t n = static_cast<size_t>(x_.numel());
    relu_grad_(dy.data<float>(), x_.data<float>(), dx_.data<float>(), n, Context::get().stream());
    return dx_;
}

// ---- Sequential ----

Tensor Sequential::forward(const Tensor& x) {
    Tensor h = x;
    for (auto& layer : layers_) {
        h = layer->forward(h);
    }
    return h;
}

Tensor Sequential::backward(const Tensor& dy) {
    Tensor g = dy;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        g = (*it)->backward(g);
    }
    return g;
}

std::vector<Parameter*> Sequential::parameters() {
    std::vector<Parameter*> out;
    for (auto& layer : layers_) {
        for (Parameter* p : layer->parameters()) {
            out.push_back(p);
        }
    }
    return out;
}

}  // namespace cudl::nn
