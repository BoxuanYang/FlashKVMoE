#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &gptq_marlin_gemm);
}
