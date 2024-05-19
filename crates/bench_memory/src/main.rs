use cudart_sys::{error::*, memory::*, stream::*};
use std::iter;

fn run_aos_3d_d(n: usize) -> CudaResult<()> {
    let stream = CudaStream::create_with_flags(cudart_sys::ffi::cudaStreamNonBlocking)?;

    let h_input: Vec<f64> = iter::repeat(1f64).take(3 * n).collect();
    let mut h_output: Vec<f64> = iter::repeat(0f64).take(3 * n).collect();

    unsafe {
        let mut d_input: DeviceBuffer<f64> = cuda_malloc_async(n * 3, &stream)?;
        let mut d_output: DeviceBuffer<f64> = cuda_malloc_async(n * 3, &stream)?;

        cuda_memcpy_h2d_async(&mut d_input, &h_input, &stream)?;

        for _ in 0..50 {
            bench_memory_cuda_kernels::cuda::bench_memory::bench_aos_3d_d(
                &d_input,
                &mut d_output,
                n,
                &stream,
            )?;
        }

        cuda_memcpy_d2h_async(&mut h_output, &d_output, &stream)?;

        cuda_free_async(d_input, &stream)?;
        cuda_free_async(d_output, &stream)?;

        stream.synchronize()?;
    }

    Ok(())
}

fn run_soa_3d_d(n: usize) -> CudaResult<()> {
    let stream = CudaStream::create_with_flags(cudart_sys::ffi::cudaStreamNonBlocking)?;

    let h_input: Vec<f64> = iter::repeat(1f64).take(3 * n).collect();
    let mut h_output: Vec<f64> = iter::repeat(0f64).take(3 * n).collect();

    unsafe {
        let mut d_input: DeviceBuffer<f64> = cuda_malloc_async(n * 3, &stream)?;
        let mut d_output: DeviceBuffer<f64> = cuda_malloc_async(n * 3, &stream)?;

        cuda_memcpy_h2d_async(&mut d_input, &h_input, &stream)?;

        for _ in 0..50 {
            bench_memory_cuda_kernels::cuda::bench_memory::bench_soa_3d_d(
                &d_input,
                &mut d_output,
                n,
                &stream,
            )?;
        }

        cuda_memcpy_d2h_async(&mut h_output, &d_output, &stream)?;

        cuda_free_async(d_input, &stream)?;
        cuda_free_async(d_output, &stream)?;

        stream.synchronize()?;
    }

    Ok(())
}

fn run_aos_4d_d(n: usize) -> CudaResult<()> {
    let stream = CudaStream::create_with_flags(cudart_sys::ffi::cudaStreamNonBlocking)?;

    let h_input: Vec<f64> = iter::repeat(1f64).take(4 * n).collect();
    let mut h_output: Vec<f64> = iter::repeat(0f64).take(4 * n).collect();

    unsafe {
        let mut d_input: DeviceBuffer<f64> = cuda_malloc_async(n * 4, &stream)?;
        let mut d_output: DeviceBuffer<f64> = cuda_malloc_async(n * 4, &stream)?;

        cuda_memcpy_h2d_async(&mut d_input, &h_input, &stream)?;

        for _ in 0..50 {
            bench_memory_cuda_kernels::cuda::bench_memory::bench_aos_4d_d(
                &d_input,
                &mut d_output,
                n,
                &stream,
            )?;
        }

        cuda_memcpy_d2h_async(&mut h_output, &d_output, &stream)?;

        cuda_free_async(d_input, &stream)?;
        cuda_free_async(d_output, &stream)?;

        stream.synchronize()?;
    }

    Ok(())
}

fn run_soa_4d_d(n: usize) -> CudaResult<()> {
    let stream = CudaStream::create_with_flags(cudart_sys::ffi::cudaStreamNonBlocking)?;

    let h_input: Vec<f64> = iter::repeat(1f64).take(4 * n).collect();
    let mut h_output: Vec<f64> = iter::repeat(0f64).take(4 * n).collect();

    unsafe {
        let mut d_input: DeviceBuffer<f64> = cuda_malloc_async(n * 4, &stream)?;
        let mut d_output: DeviceBuffer<f64> = cuda_malloc_async(n * 4, &stream)?;

        cuda_memcpy_h2d_async(&mut d_input, &h_input, &stream)?;

        for _ in 0..50 {
            bench_memory_cuda_kernels::cuda::bench_memory::bench_soa_4d_d(
                &d_input,
                &mut d_output,
                n,
                &stream,
            )?;
        }

        cuda_memcpy_d2h_async(&mut h_output, &d_output, &stream)?;

        cuda_free_async(d_input, &stream)?;
        cuda_free_async(d_output, &stream)?;

        stream.synchronize()?;
    }

    Ok(())
}

fn main() {
    let n = 1024000;

    run_aos_3d_d(n).unwrap();
    run_soa_3d_d(n).unwrap();
    run_aos_4d_d(n).unwrap();
    run_soa_4d_d(n).unwrap();
}
