#include <torch/extension.h>
#include "ConvDeepReuseForward/convDR_forward.h"
#include "ConvDeepReuseBackward/convDR_backward.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv_deep_reuse_forward", &conv_deep_reuse_forward, "DeepReuse forward");
  m.def("conv_deep_reuse_backward", &conv_deep_reuse_backward, "DeepReuse backward");
}