use std::path::PathBuf;

fn cuda_include_path() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        concat!(env!("CUDA_PATH"), "\\include")
    }

    #[cfg(target_os = "linux")]
    {
        "/usr/local/cuda/include"
    }
}

fn cuda_lib_path() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        concat!(env!("CUDA_PATH"), "\\lib\\x64")
    }

    #[cfg(target_os = "linux")]
    {
        "/usr/local/cuda/lib64"
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=wrapper.h");

    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search=native={}", cuda_lib_path());

    // Tell cargo to tell rustc to link the system libcudart
    // shared library.
    println!("cargo:rustc-link-lib=cudart");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        .clang_arg(format!("-I{}", cuda_include_path()))
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .rustified_enum("cudaError")
        .must_use_type("cudaError")
        .rustified_enum("cudaMemcpyKind")
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate cudart bindings");

    bindings
        .write_to_file(PathBuf::from("src").join("ffi").join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
