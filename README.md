# HPX/Kokkos interoperability library

WARNING: This repo is work in progress and should not be relied on for
anything. Please read the [known limitations](#known-limitations).

## What?

A header-only library for HPX/Kokkos interoperability. It provides:

- `async` versions of `Kokkos::parallel_for`, `Kokkos::parallel_reduce`, and
  `Kokkos::parallel_scan`
- `async` version of `Kokkos::deep_copy`
- HPX executors that forward work to corresponding Kokkos execution spaces
- A HPX execution policy that forwards work to corresponding Kokkos execution
  spaces for use with HPX parallel algorithms
- HPX parallel algorithm specializations for the execution policy above

## How?

This is a header-only library and does not need compiling. However, it is
recommended that it be installed using traditional CMake commands:

```
# In repository root
mkdir -p build
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=<build_type> \
    -DCMAKE_CXX_COMPILER=<compiler> \
    -DHPX_DIR=<hpx_dir> \
    -DKokkos_DIR=<kokkos_dir>
```

where `<build_type>` should be the same build type used to build HPX and
Kokkos, `<compiler>` should be the same compiler used to build HPX and Kokkos,
`<hpx_dir>` and `<kokkos_dir>` should point to the directories containing HPX
and Kokkos configuration files, respectively.

In your CMake configuration, add `find_package(HPXKokkos REQUIRED)` and link
your targets to `HPXKokkos::hpx_kokkos`. Finally, include `hpx/kokkos.hpp` in
your code.

Tests can be enabled with the CMake option `HPX_KOKKOS_ENABLE_TESTS`. All tests
can be built with the `tests` build target. The tests use `ctest`.
`HPX_KOKKOS_ENABLE_BENCHMARKS` enables benchmarks, and they can likewise be
built using the `benchmarks` target.

# Requirements

- CMake version 3.19 or newer
- HPX version 1.7.X
- Kokkos version 3.6.0 or newer
  - The build should have `Kokkos_ENABLE_HPX=ON` and
    `Kokkos_ENABLE_HPX_ASYNC_DISPATCH=ON`

For CUDA support HPX and Kokkos should be built with CUDA support. See their
respective documentation for enabling CUDA support. CUDA support requires
`Kokkos_ENABLE_CUDA_LAMBDA=ON`. The library can be used with other Kokkos
execution spaces, but only the HPX and CUDA backends are currently
asynchronous. HIP support is planned.

# API

The only supported header is `hpx/kokkos.hpp`. All other headers may change
without notice.

The following functions follow the same API as the corresponding Kokkos
functions. All execution spaces except HPX and CUDA are currently blocking and
only return ready futures.

```
namespace hpx { namespace kokkos {
hpx::shared_future<void> parallel_for_async(...);
hpx::shared_future<void> parallel_reduce_async(...);
hpx::shared_future<void> parallel_scan_async(...);
hpx::shared_future<void> deep_copy_async(...);
}}
```

The following executors correspond to Kokkos execution spaces. The executor is
only defined if the corresponding execution space is enabled in Kokkos.

```
namespace hpx { namespace kokkos {
// The following are always defined
class default_executor;
class default_host_executor;

// The following are conditionally defined
class cuda_executor;
class hip_executor;
class hpx_executor;
class openmp_executor;
class rocm_executor;
class serial_executor;
}}
```

The following execution policy can be used with parallel algorithms. It uses
the default Kokkos host execution space, unless customized with `on`.

```
namespace hpx { namespace kokkos {
static constexpr kokkos_policy kok;
}}
```

## Known issues and limitations

The following are known limitations of the library. If one of them is
particularly important for your use case, please open an issue and we'll
prioritize getting it fixed for you.

- Compilation with `nvcc` is likely not to work. Prefer `clang` for compiling
  CUDA code.
- Only the HPX and CUDA execution spaces are asynchronous. Parallel algorithms
  with other execution spaces always block and return a ready future (where
  appropriate).
- Not all HPX parallel algorithms can be used with the Kokkos executors.
  Currently the only available algorithms are `hpx::for_each`, `hpx::for_loop`,
  and `hpx::reduce`. `hpx::for_loop` only supports integer ranges (no
  iterators) and no induction or reduction objects.
- `Kokkos::View` construction and destruction (when reference count goes to
  zero) are generally blocking operations and this library does not currently
  try to solve this problem. Workarounds are: create all required views upfront
  or use unmanaged views and handle allocation and deallocation manually.
