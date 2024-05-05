use std::{marker::PhantomData, mem::MaybeUninit};

use crate::{
    error::{CudaError, CudaResult},
    stream::CudaStream,
};

#[derive(Debug)]
pub struct DeviceBuffer<T> {
    pub ptr: *mut ::std::os::raw::c_void,
    pub size_in_bytes: usize,
    _phantom: PhantomData<T>,
}

impl<T> DeviceBuffer<T> {
    pub fn new(ptr: *mut ::std::os::raw::c_void, size_in_bytes: usize) -> Self {
        return Self {
            ptr,
            size_in_bytes,
            _phantom: PhantomData,
        };
    }
}

pub unsafe fn cuda_malloc_async<T>(
    count: usize,
    stream: &CudaStream,
) -> CudaResult<DeviceBuffer<T>> {
    let size = count.checked_mul(core::mem::size_of::<T>()).unwrap_or(0);
    if size == 0 {
        return Err(CudaError(crate::ffi::cudaError::cudaErrorInvalidValue));
    }

    let mut ptr = MaybeUninit::uninit();
    crate::ffi::cudaMallocAsync(ptr.as_mut_ptr(), size, stream.as_inner()).result()?;

    Ok(DeviceBuffer::new(ptr.assume_init(), size))
}

pub unsafe fn cuda_free_async<T>(buffer: DeviceBuffer<T>, stream: &CudaStream) -> CudaResult<()> {
    crate::ffi::cudaFreeAsync(buffer.ptr, stream.as_inner()).result()
}

pub unsafe fn cuda_memcpy_h2d_async<T>(
    d_buffer: &mut DeviceBuffer<T>,
    h_buffer: &[T],
    stream: &CudaStream,
) -> CudaResult<()> {
    // TODO(gmichel): Add size check

    crate::ffi::cudaMemcpyAsync(
        d_buffer.ptr,
        h_buffer.as_ptr() as _,
        d_buffer.size_in_bytes,
        crate::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
        stream.as_inner(),
    )
    .result()
}

pub unsafe fn cuda_memcpy_d2h_async<T>(
    h_buffer: &mut [T],
    d_buffer: &DeviceBuffer<T>,
    stream: &CudaStream,
) -> CudaResult<()> {
    // TODO(gmichel): Add size check

    crate::ffi::cudaMemcpyAsync(
        h_buffer.as_ptr() as _,
        d_buffer.ptr,
        d_buffer.size_in_bytes,
        crate::ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
        stream.as_inner(),
    )
    .result()
}

pub unsafe fn cuda_memcpy_d2d_async<T>(
    d_dst: &mut DeviceBuffer<T>,
    d_src: &DeviceBuffer<T>,
    stream: &CudaStream,
) -> CudaResult<()> {
    // TODO(gmichel): Add size check

    crate::ffi::cudaMemcpyAsync(
        d_dst.ptr,
        d_src.ptr,
        d_dst.size_in_bytes,
        crate::ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
        stream.as_inner(),
    )
    .result()
}
