use std::mem::MaybeUninit;

use crate::error::{CudaError, CudaResult};

#[derive(Debug)]
pub struct CudaStream(crate::ffi::cudaStream_t);

impl CudaStream {
    pub fn create() -> CudaResult<Self> {
        let mut stream = MaybeUninit::uninit();
        unsafe {
            crate::ffi::cudaStreamCreate(stream.as_mut_ptr()).result()?;
            Ok(CudaStream(stream.assume_init()))
        }
    }

    pub fn create_with_flags(flag: u32) -> CudaResult<Self> {
        let mut stream = MaybeUninit::uninit();
        unsafe {
            crate::ffi::cudaStreamCreateWithFlags(stream.as_mut_ptr(), flag).result()?;
            Ok(CudaStream(stream.assume_init()))
        }
    }

    pub fn destroy(mut stream: Self) -> Result<(), (CudaError, Self)> {
        if stream.0.is_null() {
            return Ok(());
        }

        unsafe {
            let inner = core::mem::replace(&mut stream.0, core::ptr::null_mut());
            match crate::ffi::cudaStreamDestroy(stream.0).result() {
                Ok(()) => {
                    core::mem::forget(stream);
                    Ok(())
                }
                Err(e) => Err((e, CudaStream(inner))),
            }
        }
    }

    /// Wait until a stream's tasks are completed.
    pub fn synchronize(&self) -> CudaResult<()> {
        unsafe { crate::ffi::cudaStreamSynchronize(self.0).result() }
    }

    pub fn as_inner(&self) -> crate::ffi::cudaStream_t {
        self.0
    }
}

impl Default for CudaStream {
    fn default() -> Self {
        Self(std::ptr::null_mut())
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if self.0.is_null() {
            return;
        }

        unsafe {
            let raw_stream = core::mem::replace(&mut self.0, core::ptr::null_mut());
            crate::ffi::cudaStreamDestroy(raw_stream).result().unwrap();
        }
    }
}

impl From<CudaStream> for crate::ffi::cudaStream_t {
    fn from(stream: CudaStream) -> Self {
        stream.0
    }
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}
