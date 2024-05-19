#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_runtime.h>

#include <array>

#include "cuda_utility.h"

namespace cg = cooperative_groups;

namespace {

template <typename T, size_t D>
__global__ void benchAosKernel(const std::array<T, D>* input, std::array<T, D>* output, size_t n) {
  auto grid = cg::this_grid();
  size_t idx = grid.thread_rank();

  if (idx >= n) {
    return;
  }

  auto in = input[idx];

  std::array<T, D> out;
  for (size_t c = 0; c < D; ++c) {
    out[c] = in[c] + c;
  }

  output[idx] = out;
}

template <typename T, size_t D>
__global__ void benchSoaKernel(std::array<const T*, D> input, std::array<T*, D> output, size_t n) {
  auto grid = cg::this_grid();
  size_t idx = grid.thread_rank();

  if (idx >= n) {
    return;
  }

  std::array<T, D> in;
  for (size_t c = 0; c < D; ++c) {
    in[c] = input[c][idx];
  }

  std::array<T, D> out;
  for (size_t c = 0; c < D; ++c) {
    out[c] = in[c] + c;
  }

  for (size_t c = 0; c < D; ++c) {
    output[c][idx] = out[c];
  }
}

template <typename T, size_t D>
cudaError_t benchAos(const std::array<T, D>* input, std::array<T, D>* output, size_t n,
                     cudaStream_t stream) {
  int max_num_blocks = 0;
  int max_num_threads = 0;  // not used
  CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&max_num_blocks, &max_num_threads,
                                                benchAosKernel<T, D>, 0, 0));

  dim3 num_threads(max_num_threads, 1, 1);

  dim3 num_blocks(iDivUp(n, num_threads.x), 1, 1);

  benchAosKernel<T, D><<<num_blocks, num_threads, 0, stream>>>(input, output, n);

  return cudaGetLastError();
}

template <typename T, size_t D>
cudaError_t benchSoa(std::array<const T*, D> input, std::array<T*, D> output, size_t n,
                     cudaStream_t stream) {
  int max_num_blocks = 0;
  int max_num_threads = 0;  // not used
  CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&max_num_blocks, &max_num_threads,
                                                benchSoaKernel<T, D>, 0, 0));

  dim3 num_threads(max_num_threads, 1, 1);

  dim3 num_blocks(iDivUp(n, num_threads.x), 1, 1);

  benchSoaKernel<T, D><<<num_blocks, num_threads, 0, stream>>>(input, output, n);

  return cudaGetLastError();
}

template <typename T, size_t D>
cudaError_t benchAosWrapper(const T* input, T* output, size_t n, cudaStream_t stream) {
  return benchAos(reinterpret_cast<const std::array<T, D>*>(input),
                  reinterpret_cast<std::array<T, D>*>(output), n, stream);
}

template <typename T, size_t D>
cudaError_t benchSoaWrapper(const T* input, T* output, size_t n, cudaStream_t stream) {
  std::array<const T*, D> inputs;
  for (size_t c = 0; c < D; ++c) {
    inputs[c] = input + c * n;
  }

  std::array<T*, D> outputs;
  for (size_t c = 0; c < D; ++c) {
    outputs[c] = output + c * n;
  }

  return benchSoa(inputs, outputs, n, stream);
}
}  // namespace

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// -------------- AOS -------------------------
cudaError_t benchAos2Df(const float* input, float* output, size_t n, cudaStream_t stream) {
  return benchAosWrapper<float, 2>(input, output, n, stream);
}

cudaError_t benchAos3Df(const float* input, float* output, size_t n, cudaStream_t stream) {
  return benchAosWrapper<float, 3>(input, output, n, stream);
}

cudaError_t benchAos4Df(const float* input, float* output, size_t n, cudaStream_t stream) {
  return benchAosWrapper<float, 4>(input, output, n, stream);
}

cudaError_t benchAos2Dd(const double* input, double* output, size_t n, cudaStream_t stream) {
  return benchAosWrapper<double, 2>(input, output, n, stream);
}

cudaError_t benchAos3Dd(const double* input, double* output, size_t n, cudaStream_t stream) {
  return benchAosWrapper<double, 3>(input, output, n, stream);
}

cudaError_t benchAos4Dd(const double* input, double* output, size_t n, cudaStream_t stream) {
  return benchAosWrapper<double, 4>(input, output, n, stream);
}

// -------------- SOA -------------------------
cudaError_t benchSoa2Df(const float* input, float* output, size_t n, cudaStream_t stream) {
  return benchSoaWrapper<float, 2>(input, output, n, stream);
}

cudaError_t benchSoa3Df(const float* input, float* output, size_t n, cudaStream_t stream) {
  return benchSoaWrapper<float, 3>(input, output, n, stream);
}

cudaError_t benchSoa4Df(const float* input, float* output, size_t n, cudaStream_t stream) {
  return benchSoaWrapper<float, 4>(input, output, n, stream);
}

cudaError_t benchSoa2Dd(const double* input, double* output, size_t n, cudaStream_t stream) {
  return benchSoaWrapper<double, 2>(input, output, n, stream);
}

cudaError_t benchSoa3Dd(const double* input, double* output, size_t n, cudaStream_t stream) {
  return benchSoaWrapper<double, 3>(input, output, n, stream);
}

cudaError_t benchSoa4Dd(const double* input, double* output, size_t n, cudaStream_t stream) {
  return benchSoaWrapper<double, 4>(input, output, n, stream);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
