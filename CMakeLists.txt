# Copyright (c) 2019-2020 ETH Zurich
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13)

project(HPXKokkos CXX)

# Dependencies
find_package(HPX 1.5.0 REQUIRED)
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
add_library(hpx_kokkos
  STATIC ${PROJECT_SOURCE_DIR}/src/hpx/kokkos/execution_spaces.cpp)
target_include_directories(hpx_kokkos
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)
target_include_directories(hpx_kokkos
  INTERFACE $<INSTALL_INTERFACE:include>)
target_link_libraries(hpx_kokkos PUBLIC HPX::hpx Kokkos::kokkos)

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
  DIRECTORY src/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp")
install(
  FILES cmake/HPXKokkosConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HPXKokkos)

# Handle tests
option(HPX_KOKKOS_ENABLE_TESTS "Enable tests." OFF)
if(HPX_KOKKOS_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
