#pragma once

#include <string>
#include <vector>

#include "cudl/core/tensor.h"

namespace cudl::nn {

// Trainable tensor and its gradient (same shape) 
// backward() overwrites grad
struct Parameter {
    std::string name;
    Tensor value;
    Tensor grad;
};

// Layer-wise forward/backward
// each module saves backward needs during forward
// out buffers owned by module & reused across calls
// returned tensor is only valid until the next forward()/backward() on same module
class Module {
public:
    virtual ~Module() = default;

    virtual Tensor forward(const Tensor& x) = 0;
    virtual Tensor backward(const Tensor& dy) = 0;     // Takes dL/d(output) of latest forward(), writes param grads, returns dL/d(in)
    virtual std::vector<Parameter*> parameters() { return {}; }
};

}  // namespace cudl::nn
