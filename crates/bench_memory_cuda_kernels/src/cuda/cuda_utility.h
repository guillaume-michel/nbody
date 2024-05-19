#pragma once

#include <cuda_runtime.h>

#include <type_traits>

/**
 * @brief iDivUp
 */
template <typename I0, typename I1>
inline __device__ __host__ auto iDivUp(I0 a, I1 b) -> std::common_type_t<I0, I1> {
  using I = std::common_type_t<I0, I1>;
  return I(a + b - 1) / I(b);
}

/**
 * cudaCheckError
 * @ingroup util
 */
inline cudaError_t cudaCheckError(cudaError_t retval, const char* /*txt*/, const char* file,
                                  int line) {
  if (retval != cudaSuccess) {
      //std::cerr << file << ":" << line << ": cuda error: " << cudaGetErrorString(retval) << std::endl;
  }

  return retval;
}

#define CHECK_CUDA(ans) cudaAssert((ans), __FILE__, __LINE__)
inline void cudaAssert(cudaError_t status, const char* file, int line) {
  if (cudaCheckError(status, "", file, line) != cudaSuccess) {
    abort();
  }
}
