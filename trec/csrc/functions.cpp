#include "ConvDeepReuseBackward/convDR_backward.h"
#include "ConvDeepReuseForward/convDR_forward.h"
#include <torch/extension.h>

// Registers _C as an extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("conv_deep_reuse_forward", conv_deep_reuse_forward);
    m.def("conv_deep_reuse_backward", conv_deep_reuse_backward);
}

// Defines the operators
TORCH_LIBRARY(TORCH_EXTENSION_NAME, m)
{
    m.def("conv_deep_reuse_forward", conv_deep_reuse_forward);
    m.def("conv_deep_reuse_backward", conv_deep_reuse_backward);
}