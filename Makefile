arch = skylake
N = 50000000

all: bench

nbody_rust: nbody.rs
	rustc -C opt-level=3 -C target-cpu=$(arch) --C codegen-units=1 -C llvm-args='-unroll-threshold=500' $^ -o $@

nbody_cpp: nbody.cpp
	clang++-11 -pipe -Ofast -fomit-frame-pointer -march=$(arch) -mllvm -unroll-threshold=500 $^ -o $@

nbody_lisp: nbody.lisp
	sbcl --userinit /dev/null --eval '(load (compile-file "$^"))' --eval '(save-lisp-and-die "$@" :purify t)'

clean:
	rm -rf nbody_rust nbody_cpp nbody_lisp nbody.fasl

bench: nbody_rust nbody_cpp nbody_lisp
	@echo "-----------RUST----------------"
	@time ./nbody_rust $(N)

	@echo "-----------C++-----------------"
	@time ./nbody_cpp $(N)

	@echo "-----------LISP----------------"
	@time sbcl --dynamic-space-size 500 --noinform --core nbody_lisp --userinit /dev/null --eval '(time (main))' --eval '(quit)' $(N)
