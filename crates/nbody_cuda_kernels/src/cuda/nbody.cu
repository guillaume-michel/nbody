#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <array>

#include "cuda_utility.h"

namespace cg = cooperative_groups;

namespace {

template <typename T>
__constant__ T softening_squared;

template <typename T>
cudaError_t setSofteningSquared(T softeningSq, cudaStream_t stream) {
  return cudaMemcpyToSymbolAsync(softening_squared<T>, &softeningSq, sizeof(T), 0,
                                 cudaMemcpyHostToDevice, stream);
}

template <typename T>
__device__ T rsqrt_T(T x);

template <>
__device__ float rsqrt_T<float>(float x) {
  return rsqrtf(x);
}

template <>
__device__ double rsqrt_T<double>(double x) {
  return rsqrt(x);
}

template <typename T, size_t N>
__device__ std::array<T, N> bodyBodyInteraction(std::array<T, N> ai, const std::array<T, N>& bi,
                                                const std::array<T, N + 1>& bj) {
  // r_ij  [3 FLOPS]
  std::array<T, N> r;
  //#pragma unroll
  for (size_t k = 0; k < N; ++k) {
    r[k] = bj[k] - bi[k];
  }

  // dist_sqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
  // const T dist_sqr = r.x * r.x + r.y * r.y + r.z * r.z + getSofteningSquared<T>();

  T dist_sqr = T(0);
  //#pragma unroll
  for (size_t k = 0; k < N; ++k) {
    dist_sqr += r[k] * r[k];
  }
  dist_sqr += softening_squared<T>;

  // inv_dist_cube =1/dist_sqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
  const T inv_dist = rsqrt_T(dist_sqr);
  const T inv_dist_cube = inv_dist * inv_dist * inv_dist;

  // s = m_j * invDistCube [1 FLOP]
  const T s = bj[N] * inv_dist_cube;

  // a_i =  a_i + s * r_ij [6 FLOPS]
  //#pragma unroll
  for (size_t k = 0; k < N; ++k) {
    ai[k] += r[k] * s;
  }

  return ai;
}

template <typename T, size_t N>
__device__ std::array<T, N> computeBodyAccel(const std::array<T, N>& body_position,
                                             const std::array<T, N>* positions, const T* masses,
                                             unsigned int numTiles, cg::thread_block cta) {
  extern __shared__ std::array<T, N + 1> shared_pos[];

  std::array<T, N> acc = {T(0)};

  for (int tile = 0; tile < numTiles; tile++) {
    {
      auto index = tile * blockDim.x + threadIdx.x;
      auto p = positions[index];
      //#pragma unroll
      for (size_t k = 0; k < N; ++k) {
        shared_pos[threadIdx.x][k] = p[k];
      }
      shared_pos[threadIdx.x][N] = masses[index];
    }
    cg::sync(cta);

// This is the "tile_calculation" from the GPUG3 article.
    //#pragma unroll 128
    for (unsigned int counter = 0; counter < blockDim.x; ++counter) {
      acc = bodyBodyInteraction(acc, body_position, shared_pos[counter]);
    }

    cg::sync(cta);
  }

  return acc;
}

template <typename T, size_t N>
__global__ void integrateBodiesKernel(std::array<T, N>* __restrict__ output_positions,
                                      const std::array<T, N>* __restrict__ input_positions,
                                      std::array<T, N>* __restrict__ velocities, const T* masses,
                                      unsigned int num_bodies, T delta_time, T damping,
                                      unsigned int num_tiles) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  // FIXEME: do not exit early when we have block synchronization
  if (index >= num_bodies) {
    return;
  }

  auto position = input_positions[index];

  auto accel = computeBodyAccel(position, input_positions, masses, num_tiles, cta);

  // acceleration = force / mass;
  // new velocity = old velocity + acceleration * deltaTime
  // note we factor out the body's mass from the equation, here and in
  // bodyBodyInteraction
  // (because they cancel out).  Thus here force == acceleration
  auto velocity = velocities[index];

  //#pragma unroll
  for (size_t k = 0; k < N; ++k) {
    velocity[k] += accel[k] * delta_time;
  }

  //#pragma unroll
  for (size_t k = 0; k < N; ++k) {
    velocity[k] *= damping;
  }

  // new position = old position + velocity * deltaTime
  //#pragma unroll
  for (size_t k = 0; k < N; ++k) {
    position[k] += velocity[k] * delta_time;
  }

  // store new position and velocity
  output_positions[index] = position;
  velocities[index] = velocity;
}

template <typename T, size_t N>
cudaError_t integrateNbodySystem(const std::array<T, N>* input_positions,
                                 std::array<T, N>* velocities, const T* masses, size_t num_bodies,
                                 std::array<T, N>* output_positions, T delta_time, T damping,
                                 size_t block_size, cudaStream_t stream) {
  size_t num_blocks = (num_bodies + block_size - 1) / block_size;
  size_t num_tiles = (num_bodies + block_size - 1) / block_size;
  size_t shared_mem_size = block_size * (N + 1) * sizeof(T);  // N floats for pos + 1 float for mass

  integrateBodiesKernel<T, N><<<num_blocks, block_size, shared_mem_size, stream>>>(
      output_positions, input_positions, velocities, masses, num_bodies, delta_time, damping,
      num_tiles);

  return cudaGetLastError();
}

template <typename T, size_t N>
cudaError_t integrateNbodySystem(const T* input_positions, T* velocities, const T* masses,
                                 size_t num_bodies, T* output_positions, T delta_time, T damping,
                                 size_t block_size, cudaStream_t stream) {
  return integrateNbodySystem(reinterpret_cast<const std::array<T, N>*>(input_positions),
                              reinterpret_cast<std::array<T, N>*>(velocities), masses, num_bodies,
                              reinterpret_cast<std::array<T, N>*>(output_positions), delta_time,
                              damping, block_size, stream);
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
// cudaError_t setSofteningSquaredF32(float softeningSq) { return setSofteningSquared(softeningSq);
// }

cudaError_t setSofteningSquaredF64(double softeningSq, cudaStream_t stream) {
  return setSofteningSquared(softeningSq, stream);
}

// cudaError_t integrateNbodySystem2DF32(const float* input_positions, float* velocities,
//                                       const float* masses, size_t num_bodies,
//                                       float* output_positions, float delta_time, float damping,
//                                       size_t block_size, cudaStream_t stream) {
//   return integrateNbodySystem<float, 2>(input_positions, velocities, masses, num_bodies,
//                                         output_positions, delta_time, damping, block_size,
//                                         stream);
// }

// cudaError_t integrateNbodySystem3DF32(const float* input_positions, float* velocities,
//                                       const float* masses, size_t num_bodies,
//                                       float* output_positions, float delta_time, float damping,
//                                       size_t block_size, cudaStream_t stream) {
//   return integrateNbodySystem<float, 3>(input_positions, velocities, masses, num_bodies,
//                                         output_positions, delta_time, damping, block_size,
//                                         stream);
// }

// cudaError_t integrateNbodySystem2DF64(const double* input_positions, double* velocities,
//                                       const double* masses, size_t num_bodies,
//                                       double* output_positions, double delta_time, double
//                                       damping, size_t block_size, cudaStream_t stream) {
//   return integrateNbodySystem<double, 2>(input_positions, velocities, masses, num_bodies,
//                                          output_positions, delta_time, damping, block_size,
//                                          stream);
// }

cudaError_t integrateNbodySystem3DF64(const double* input_positions, double* velocities,
                                      const double* masses, size_t num_bodies,
                                      double* output_positions, double delta_time, double damping,
                                      size_t block_size, cudaStream_t stream) {
  return integrateNbodySystem<double, 3>(input_positions, velocities, masses, num_bodies,
                                         output_positions, delta_time, damping, block_size, stream);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
