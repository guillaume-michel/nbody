use core::ffi::CStr;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CudaError(pub crate::ffi::cudaError);

/// Result type for most CUDA functions.
pub type CudaResult<T> = Result<T, CudaError>;

impl CudaError {
    /// Gets the name for this error.
    pub fn error_name(&self) -> &CStr {
        unsafe { CStr::from_ptr(crate::ffi::cudaGetErrorName(self.0)) }
    }

    /// Gets the error string for this error.
    pub fn error_string(&self) -> &CStr {
        unsafe { CStr::from_ptr(crate::ffi::cudaGetErrorString(self.0)) }
    }
}

impl std::fmt::Debug for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let err_str = self.error_string();
        f.debug_tuple("CudaError")
            .field(&self.0)
            .field(&err_str)
            .finish()
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CudaError {}

impl crate::ffi::cudaError {
    #[inline]
    pub fn result(self) -> CudaResult<()> {
        match self {
            crate::ffi::cudaError::cudaSuccess => Ok(()),
            _ => Err(CudaError(self)),
        }
    }
}

impl From<crate::ffi::cudaError> for CudaResult<()> {
    fn from(status: crate::ffi::cudaError) -> Self {
        status.result()
    }
}
