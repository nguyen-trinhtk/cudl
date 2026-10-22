#include "cudl/nn/loss.h"

#include "buffers.h"
#include "cudl/core/context.h"
#include "cudl/core/nvtx.h"
#include "cudl/kernels/registry.h"

namespace cudl::nn {

using detail::ensure;

CrossEntropyLoss::CrossEntropyLoss()
    : sum_(Tensor::empty({1})),
      loss_(Tensor::empty({1})),
      workspace_(Tensor::empty({static_cast<int64_t>(kernels::kReduceWorkspaceFloats)})) {
    cross_entropy_ = kernels::registry<kernels::CrossEntropyLossFn>().get_default("cross_entropy_loss").fn;
    // A tree reduction rather than the first-registered atomic one, so the loss is deterministic
    sum_all_ = kernels::registry<kernels::ReduceAllFn>().get_default("sum", "warp").fn;
    scale_ = kernels::registry<kernels::ScaleFn>().get_default("scale").fn;
}

Tensor CrossEntropyLoss::forward(const Tensor& logits, const Tensor& labels) {
    CUDL_NVTX_RANGE("cross_entropy.forward");
    CUDL_CHECK(logits.dim() == 2, "logits must be [batch, classes]");
    detail::check_matrix(logits, logits.shape()[1], "logits");
    const int64_t batch = logits.shape()[0];
    CUDL_CHECK(labels.defined() && labels.dtype() == DType::I32 && labels.device() == Device::CUDA,
               "labels must be an i32 GPU tensor");
    CUDL_CHECK(labels.dim() == 1 && labels.shape()[0] == batch, "labels must be [" << batch << "]");
    CUDL_CHECK(batch > 0, "empty batch");
    ensure(row_loss_, {batch});
    ensure(dlogits_, logits.shape());

    const cudaStream_t stream = Context::get().stream();
    const size_t rows = static_cast<size_t>(batch);
    const size_t cols = static_cast<size_t>(logits.shape()[1]);
    const float inv_batch = 1.0f / static_cast<float>(batch);
    cross_entropy_(logits.data<float>(), labels.data<int32_t>(), row_loss_.data<float>(), dlogits_.data<float>(), rows,
                   cols, inv_batch, stream);
    sum_all_(row_loss_.data<float>(), sum_.data<float>(), rows, workspace_.data<float>(), stream);
    scale_(sum_.data<float>(), inv_batch, loss_.data<float>(), 1, stream);
    return loss_;
}

Tensor CrossEntropyLoss::backward() const {
    CUDL_CHECK(dlogits_.defined(), "CrossEntropyLoss::backward called before forward");
    return dlogits_;
}

}  // namespace cudl::nn
