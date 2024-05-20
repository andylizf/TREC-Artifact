#pragma once
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
[[__gnu__::__always_inline__]] inline void CHECK_SHAPE(const at::Tensor& tensor, const at::IntArrayRef shape)
{
    TORCH_CHECK(tensor.sizes() == shape, "Expected shape ", shape, " but got ", tensor.sizes());
}

[[gnu::always_inline]] [[gnu::hot]]
inline double timestamp()
{
#ifndef NDEBUG
    timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + 1e-6 * t.tv_usec;
#endif
    return 0; // should be optimized out
}
#define TIMER_START TIMER_t = timestamp()

template <typename String>
[[gnu::always_inline]]
inline void TIMER_LAP_impl([[maybe_unused]] String str, [[maybe_unused]] double& TIMER_t)
{
#ifndef NDEBUG
    torch::cuda::synchronize();
    cudaDeviceSynchronize();
    printf("%s: %f\n", str, timestamp() - TIMER_t);
    TIMER_t = timestamp();
#endif
}
#define TIMER_LAP(str) TIMER_LAP_impl(str, TIMER_t)

template <typename... Args>
[[gnu::always_inline]]
inline void DEBUG_PRINT([[maybe_unused]] const Args&... args)
{
#ifndef NDEBUG
    printf(args...);
#endif
}