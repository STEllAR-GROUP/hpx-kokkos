cmake_minimum_required(VERSION 3.3.2)

project(kokkos_hpx_interop CXX)

find_package(HPX REQUIRED)
find_package(Kokkos REQUIRED)

# Check that Kokkos and HPX options are consistent.
if(Kokkos_ENABLE_CUDA)
  kokkos_check(OPTIONS CUDA_LAMBDA)
  if(NOT HPX_WITH_CUDA)
    message(FATAL_ERROR "Kokkos was built with CUDA support, HPX was not")
  endif()
else()
  if(HPX_WITH_CUDA)
    message(FATAL_ERROR "HPX was built with CUDA support, Kokkos was not")
  endif()
endif()

kokkos_check(DEVICES HPX)
kokkos_check(OPTIONS HPX_ASYNC_DISPATCH)

# Main target provided by this project
add_library(kokkos_hpx_interop INTERFACE)
target_include_directories(kokkos_hpx_interop INTERFACE ${PROJECT_SOURCE_DIR}/include)

# Examples using this project
add_hpx_executable(kokkos_hpx SOURCES src/kokkos_hpx.cpp)
target_link_libraries(kokkos_hpx PRIVATE Kokkos::kokkos kokkos_hpx_interop)
