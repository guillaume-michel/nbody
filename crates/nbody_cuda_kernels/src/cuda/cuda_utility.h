#pragma once

#include <type_traits>

/**
 * @brief iDivUp
 */
template <typename I0, typename I1>
inline __device__ __host__ auto iDivUp(I0 a, I1 b) -> std::common_type_t<I0, I1> {
  using I = std::common_type_t<I0, I1>;
  return I(a + b - 1) / I(b);
}
