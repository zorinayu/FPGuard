#include <torch/extension.h>

std::vector<at::Tensor> rowwise_topk_cuda(const at::Tensor &scores, int64_t k);

std::vector<at::Tensor> rowwise_topk(const at::Tensor &scores, int64_t k) {
  TORCH_CHECK(scores.is_cuda(), "scores must be CUDA tensor");
  TORCH_CHECK(scores.dim() == 2, "scores must be 2D [Q, K]");
  TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");
  TORCH_CHECK(k > 0, "k must be positive");
  TORCH_CHECK(k <= scores.size(1), "k must be <= number of columns (K)");
  TORCH_CHECK(scores.scalar_type() == at::kFloat || scores.scalar_type() == at::kDouble,
              "scores dtype must be float32 or float64");
  return rowwise_topk_cuda(scores, k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rowwise_topk", &rowwise_topk, "Row-wise TopK (CUDA)");
}

