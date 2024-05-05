use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-lineinfo")
        .arg("--extended-lambda")
        .watch(vec!["src/cuda/"]);

    // Write lib to the $OUT_DIR directory.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    builder.build_lib(out_path.join("libnbody_kernels.a"));

    println!("cargo:rustc-link-search=native={}", out_path.display());
    println!("cargo:rustc-link-lib=nbody_kernels");
}
