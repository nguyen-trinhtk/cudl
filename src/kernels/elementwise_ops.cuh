#pragma once

// Functor: Use struct + operator() to call like function
// while compatible with templated kernel launch

namespace cudl::kernels::ops {
// Unary
struct Neg {
    __device__ float operator()(float x) const { return -x; }
};
struct Relu {
    __device__ float operator()(float x) const { return x < 0.0f ? 0.0f : x; } // x < 0 rather than fmaxf so NaN propagates
};
struct Exp {
    __device__ float operator()(float x) const { return expf(x); }
};
struct Tanh {
    __device__ float operator()(float x) const { return tanhf(x); }
};
struct Sigmoid {
    __device__ float operator()(float x) const { return 1.0f / (1.0f + expf(-x)); }
};
struct Scale {
    float alpha;
    __device__ float operator()(float x) const { return alpha * x; }
};

// Binary
struct Add {
    __device__ float operator()(float a, float b) const { return a + b; }
};
struct Sub {
    __device__ float operator()(float a, float b) const { return a - b; }
};
struct Mul {
    __device__ float operator()(float a, float b) const { return a * b; }
};
struct Div {
    __device__ float operator()(float a, float b) const { return a / b; }
};
}  // namespace cudl::kernels::ops
