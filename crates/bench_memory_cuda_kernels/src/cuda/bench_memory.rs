mod ffi {
    extern "C" {
        pub fn benchAos2Df(
            input: *const f32,
            output: *mut f32,
            n: usize,
            stream: cudart_sys::ffi::cudaStream_t,
        ) -> cudart_sys::ffi::cudaError;

        pub fn benchAos3Df(
            input: *const f32,
            output: *mut f32,
            n: usize,
            stream: cudart_sys::ffi::cudaStream_t,
        ) -> cudart_sys::ffi::cudaError;

        pub fn benchAos4Df(
            input: *const f32,
            output: *mut f32,
            n: usize,
            stream: cudart_sys::ffi::cudaStream_t,
        ) -> cudart_sys::ffi::cudaError;

        pub fn benchAos2Dd(
            input: *const f64,
            output: *mut f64,
            n: usize,
            stream: cudart_sys::ffi::cudaStream_t,
        ) -> cudart_sys::ffi::cudaError;

        pub fn benchAos3Dd(
            input: *const f64,
            output: *mut f64,
            n: usize,
            stream: cudart_sys::ffi::cudaStream_t,
        ) -> cudart_sys::ffi::cudaError;

        pub fn benchAos4Dd(
            input: *const f64,
            output: *mut f64,
            n: usize,
            stream: cudart_sys::ffi::cudaStream_t,
        ) -> cudart_sys::ffi::cudaError;

        pub fn benchSoa2Df(
            input: *const f32,
            output: *mut f32,
            n: usize,
            stream: cudart_sys::ffi::cudaStream_t,
        ) -> cudart_sys::ffi::cudaError;

        pub fn benchSoa3Df(
            input: *const f32,
            output: *mut f32,
            n: usize,
            stream: cudart_sys::ffi::cudaStream_t,
        ) -> cudart_sys::ffi::cudaError;

        pub fn benchSoa4Df(
            input: *const f32,
            output: *mut f32,
            n: usize,
            stream: cudart_sys::ffi::cudaStream_t,
        ) -> cudart_sys::ffi::cudaError;

        pub fn benchSoa2Dd(
            input: *const f64,
            output: *mut f64,
            n: usize,
            stream: cudart_sys::ffi::cudaStream_t,
        ) -> cudart_sys::ffi::cudaError;

        pub fn benchSoa3Dd(
            input: *const f64,
            output: *mut f64,
            n: usize,
            stream: cudart_sys::ffi::cudaStream_t,
        ) -> cudart_sys::ffi::cudaError;

        pub fn benchSoa4Dd(
            input: *const f64,
            output: *mut f64,
            n: usize,
            stream: cudart_sys::ffi::cudaStream_t,
        ) -> cudart_sys::ffi::cudaError;
    }
}

// ------------------- AOS ---------------------------------
pub fn bench_aos_2d_f(
    input: &cudart_sys::memory::DeviceBuffer<f32>,
    output: &mut cudart_sys::memory::DeviceBuffer<f32>,
    n: usize,
    stream: &cudart_sys::stream::CudaStream,
) -> cudart_sys::error::CudaResult<()> {
    unsafe { ffi::benchAos2Df(input.ptr as _, output.ptr as _, n, stream.as_inner()).result() }
}

pub fn bench_aos_3d_f(
    input: &cudart_sys::memory::DeviceBuffer<f32>,
    output: &mut cudart_sys::memory::DeviceBuffer<f32>,
    n: usize,
    stream: &cudart_sys::stream::CudaStream,
) -> cudart_sys::error::CudaResult<()> {
    unsafe { ffi::benchAos3Df(input.ptr as _, output.ptr as _, n, stream.as_inner()).result() }
}

pub fn bench_aos_4d_f(
    input: &cudart_sys::memory::DeviceBuffer<f32>,
    output: &mut cudart_sys::memory::DeviceBuffer<f32>,
    n: usize,
    stream: &cudart_sys::stream::CudaStream,
) -> cudart_sys::error::CudaResult<()> {
    unsafe { ffi::benchAos4Df(input.ptr as _, output.ptr as _, n, stream.as_inner()).result() }
}

pub fn bench_aos_2d_d(
    input: &cudart_sys::memory::DeviceBuffer<f64>,
    output: &mut cudart_sys::memory::DeviceBuffer<f64>,
    n: usize,
    stream: &cudart_sys::stream::CudaStream,
) -> cudart_sys::error::CudaResult<()> {
    unsafe { ffi::benchAos2Dd(input.ptr as _, output.ptr as _, n, stream.as_inner()).result() }
}

pub fn bench_aos_3d_d(
    input: &cudart_sys::memory::DeviceBuffer<f64>,
    output: &mut cudart_sys::memory::DeviceBuffer<f64>,
    n: usize,
    stream: &cudart_sys::stream::CudaStream,
) -> cudart_sys::error::CudaResult<()> {
    unsafe {
        crate::cuda::bench_memory::ffi::benchAos3Dd(
            input.ptr as _,
            output.ptr as _,
            n,
            stream.as_inner(),
        )
        .result()
    }
}

pub fn bench_aos_4d_d(
    input: &cudart_sys::memory::DeviceBuffer<f64>,
    output: &mut cudart_sys::memory::DeviceBuffer<f64>,
    n: usize,
    stream: &cudart_sys::stream::CudaStream,
) -> cudart_sys::error::CudaResult<()> {
    unsafe { ffi::benchAos4Dd(input.ptr as _, output.ptr as _, n, stream.as_inner()).result() }
}

// ------------------- SOA ---------------------------------
pub fn bench_soa_2d_f(
    input: &cudart_sys::memory::DeviceBuffer<f32>,
    output: &mut cudart_sys::memory::DeviceBuffer<f32>,
    n: usize,
    stream: &cudart_sys::stream::CudaStream,
) -> cudart_sys::error::CudaResult<()> {
    unsafe { ffi::benchSoa2Df(input.ptr as _, output.ptr as _, n, stream.as_inner()).result() }
}

pub fn bench_soa_3d_f(
    input: &cudart_sys::memory::DeviceBuffer<f32>,
    output: &mut cudart_sys::memory::DeviceBuffer<f32>,
    n: usize,
    stream: &cudart_sys::stream::CudaStream,
) -> cudart_sys::error::CudaResult<()> {
    unsafe { ffi::benchSoa3Df(input.ptr as _, output.ptr as _, n, stream.as_inner()).result() }
}

pub fn bench_soa_4d_f(
    input: &cudart_sys::memory::DeviceBuffer<f32>,
    output: &mut cudart_sys::memory::DeviceBuffer<f32>,
    n: usize,
    stream: &cudart_sys::stream::CudaStream,
) -> cudart_sys::error::CudaResult<()> {
    unsafe { ffi::benchSoa4Df(input.ptr as _, output.ptr as _, n, stream.as_inner()).result() }
}

pub fn bench_soa_2d_d(
    input: &cudart_sys::memory::DeviceBuffer<f64>,
    output: &mut cudart_sys::memory::DeviceBuffer<f64>,
    n: usize,
    stream: &cudart_sys::stream::CudaStream,
) -> cudart_sys::error::CudaResult<()> {
    unsafe { ffi::benchSoa2Dd(input.ptr as _, output.ptr as _, n, stream.as_inner()).result() }
}

pub fn bench_soa_3d_d(
    input: &cudart_sys::memory::DeviceBuffer<f64>,
    output: &mut cudart_sys::memory::DeviceBuffer<f64>,
    n: usize,
    stream: &cudart_sys::stream::CudaStream,
) -> cudart_sys::error::CudaResult<()> {
    unsafe { ffi::benchSoa3Dd(input.ptr as _, output.ptr as _, n, stream.as_inner()).result() }
}

pub fn bench_soa_4d_d(
    input: &cudart_sys::memory::DeviceBuffer<f64>,
    output: &mut cudart_sys::memory::DeviceBuffer<f64>,
    n: usize,
    stream: &cudart_sys::stream::CudaStream,
) -> cudart_sys::error::CudaResult<()> {
    unsafe { ffi::benchSoa4Dd(input.ptr as _, output.ptr as _, n, stream.as_inner()).result() }
}
