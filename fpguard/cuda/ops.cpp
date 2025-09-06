#include <torch/extension.h>

std::vector<at::Tensor> rowwise_topk_cuda(const at::Tensor &scores, int64_t k);

std::vector<at::Tensor> rowwise_topk(const at::Tensor &scores, int64_t k) {
  TORCH_CHECK(scores.is_cuda(), "scores must be CUDA tensor");
  return rowwise_topk_cuda(scores, k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rowwise_topk", &rowwise_topk, "Row-wise TopK (CUDA)");
}

