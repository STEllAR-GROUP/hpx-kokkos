cmake_minimum_required(VERSION 3.13)

project(HPXKokkos CXX)

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
add_library(hpx_kokkos INTERFACE)
target_include_directories(hpx_kokkos
  INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)
target_link_libraries(hpx_kokkos INTERFACE HPX::hpx Kokkos::kokkos)

include(GNUInstallDirs)
install(
  TARGETS hpx_kokkos
  EXPORT HPXKokkos)
install(
  EXPORT HPXKokkos
  FILE HPXKokkosTargets.cmake
  NAMESPACE HPXKokkos::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HPXKokkos)
install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
  FILES cmake/HPXKokkosConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HPXKokkos)

# Examples using this project
add_executable(hpx_kokkos_test EXCLUDE_FROM_ALL src/hpx_kokkos_test.cpp)
target_link_libraries(hpx_kokkos_test PRIVATE hpx_kokkos)
