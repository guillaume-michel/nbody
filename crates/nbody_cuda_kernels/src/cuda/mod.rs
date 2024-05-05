// #[allow(non_upper_case_globals)]
// #[allow(non_camel_case_types)]
// #[allow(non_snake_case)]
// #[allow(dead_code)]
mod ffi {
    extern "C" {
        #[doc = "Integrate N-Body 3D system with double precision in CUDA"]
        pub fn integrateNbodySystem3DF64(
            input_positions: *const f64,
            velocities: *mut f64,
            num_bodies: usize,
            output_positions: *mut f64,
            delta_time: f64,
            damping: f64,
            block_size: usize,
            stream: cudart_sys::ffi::cudaStream_t,
        ) -> cudart_sys::ffi::cudaError;

        //cudaError_t setSofteningSquaredF64(double softeningSq) { return setSofteningSquared(softeningSq); }
        #[doc = "Set softening squared factor"]
        pub fn setSofteningSquaredF64(
            softeningSq: f64,
            stream: cudart_sys::ffi::cudaStream_t,
        ) -> cudart_sys::ffi::cudaError;
    }
}

pub fn integrate_nbody_system_3d_f64(
    input_positions: &cudart_sys::memory::DeviceBuffer<f64>,
    velocities: &mut cudart_sys::memory::DeviceBuffer<f64>,
    num_bodies: usize,
    output_positions: &mut cudart_sys::memory::DeviceBuffer<f64>,
    delta_time: f64,
    damping: f64,
    block_size: usize,
    stream: &cudart_sys::stream::CudaStream,
) -> cudart_sys::error::CudaResult<()> {
    unsafe {
        crate::cuda::ffi::integrateNbodySystem3DF64(
            input_positions.ptr as _,
            velocities.ptr as _,
            num_bodies,
            output_positions.ptr as _,
            delta_time,
            damping,
            block_size,
            stream.as_inner(),
        )
        .result()
    }
}

pub fn set_softening_squared_f64(
    softening_sq: f64,
    stream: &cudart_sys::stream::CudaStream,
) -> cudart_sys::error::CudaResult<()> {
    unsafe { crate::cuda::ffi::setSofteningSquaredF64(softening_sq, stream.as_inner()).result() }
}
