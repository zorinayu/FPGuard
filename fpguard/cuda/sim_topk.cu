#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Simple row-wise top-k using partial selection per row.
// scores: [Q, K]

template <typename scalar_t>
__global__ void rowwise_topk_kernel(
    const scalar_t* __restrict__ scores,
    scalar_t* __restrict__ topk_vals,
    int64_t* __restrict__ topk_idx,
    const int Q,
    const int K,
    const int k) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= Q) return;

  // Pointers to this row
  const scalar_t* row_ptr = scores + row * K;

  // Simple O(K * k) partial selection
  for (int i = 0; i < k; ++i) {
    topk_vals[row * k + i] = (scalar_t)-1e30;
    topk_idx[row * k + i] = -1;
  }

  for (int j = 0; j < K; ++j) {
    scalar_t v = row_ptr[j];
    // find position to insert
    int pos = -1;
    if (v > topk_vals[row * k + (k - 1)]) {
      pos = k - 1;
      while (pos > 0 && v > topk_vals[row * k + (pos - 1)]) {
        topk_vals[row * k + pos] = topk_vals[row * k + (pos - 1)];
        topk_idx[row * k + pos] = topk_idx[row * k + (pos - 1)];
        --pos;
      }
      topk_vals[row * k + pos] = v;
      topk_idx[row * k + pos] = j;
    }
  }
}

std::vector<at::Tensor> rowwise_topk_cuda(const at::Tensor &scores, int64_t k) {
  TORCH_CHECK(scores.is_cuda(), "scores must be CUDA tensor");
  TORCH_CHECK(scores.dim() == 2, "scores must be 2D [Q, K]");
  const auto Q = scores.size(0);
  const auto K = scores.size(1);

  auto opts_val = scores.options();
  auto opts_idx = scores.options().dtype(at::kLong);
  auto topk_vals = at::empty({Q, k}, opts_val);
  auto topk_idx = at::empty({Q, k}, opts_idx);

  const int threads = 256;
  const int blocks = (Q + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(scores.scalar_type(), "rowwise_topk_cuda", ([&] {
    rowwise_topk_kernel<scalar_t><<<blocks, threads>>>(
        scores.data_ptr<scalar_t>(),
        topk_vals.data_ptr<scalar_t>(),
        topk_idx.data_ptr<int64_t>(),
        (int)Q, (int)K, (int)k);
  }));

  return {topk_vals, topk_idx};
}


