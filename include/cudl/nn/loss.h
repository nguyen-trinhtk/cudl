#pragma once

#include "cudl/core/tensor.h"
#include "cudl/kernels/elementwise.h"
#include "cudl/kernels/loss.h"
#include "cudl/kernels/reduce.h"

namespace cudl::nn {

// Mean fused softmax/CE over a batch
// backward() only return what forward() already computed
class CrossEntropyLoss {
public:
    CrossEntropyLoss();

    // logits [batch, classes] (F32), labels [batch] (I32, values in [0, classes))
    Tensor forward(const Tensor& logits, const Tensor& labels);
    // dL/dlogits [batch, classes] of the latest forward
    Tensor backward() const;

private:
    Tensor row_loss_;   // [batch]
    Tensor sum_;        // [1]
    Tensor loss_;       // [1]
    Tensor dlogits_;    // [batch, classes]
    Tensor workspace_;  // reduction partials
    kernels::CrossEntropyLossFn cross_entropy_;
    kernels::ReduceAllFn sum_all_;
    kernels::ScaleFn scale_;
};

}  // namespace cudl::nn
