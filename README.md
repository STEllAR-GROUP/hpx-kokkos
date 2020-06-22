# HPX/Kokkos interoperability library

WARNING: This repo is work in progress and should not be relied on for
anything.

## What?

A header-only library for HPX/Kokkos interoperability. It provides:

- `async` versions of `Kokkos::parallel_for`, `parallel_reduce`, and `parallel_scan`
- `async` version of `Kokkos::deep_copy`
- Kokkos executors that forward work to corresponding Kokkos execution spaces

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

# Requirements

- CMake version 3.13 or newer
- HPX version 1.5.0 or newer
- Kokkos version 3.1.0 or newer (TODO: check which version has HPX updates)
  - The build should have `Kokkos_ENABLE_HPX_ASYNC_DISPATCH=ON`

For CUDA support HPX and Kokkos should be built with CUDA support. See their
respective documentation for enabling CUDA support. The library can be used
with other Kokkos execution spaces, but only the HPX and CUDA backends are
currently asynchronous. HIP support is planned.

# API

The only supported header is `hpx/kokkos.hpp`. All other headers may change
without notice.

The following follow the same API as the corresponding Kokkos functions. All
execution spaces except HPX and CUDA are currently blocking and only return
ready futures.

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

The following is a helper function for creating execution spaces that are
independent. It is allowed to return the same execution space instance on
subsequent invocations.

```
namespace hpx { namespace kokkos {
template <typename ExecutionSpace>
ExecutionSpace make_execution_space();
}}
```

## Known limitations

- Compilation with `nvcc` is likely not to work. Prefer `clang` for compiling
  CUDA code.
- Only the HPX and CUDA execution spaces are asynchronous
- Not all HPX parallel algorithms can be used with the Kokkos executors;
  currently tested algorithms are:
  - `hpx::for_each`
- The Kokkos executors do not support continuations (`then_execute` and `bulk_then_execute`)
