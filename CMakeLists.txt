cmake_minimum_required(VERSION 3.3.2)

project(kokkos_hpx_interop CXX)

find_package(HPX REQUIRED)
find_package(Kokkos REQUIRED)

# Main target provided by this project
add_library(kokkos_hpx_interop INTERFACE)
target_include_directories(kokkos_hpx_interop INTERFACE ${PROJECT_SOURCE_DIR}/include)

# Examples using this project
add_hpx_executable(kokkos_hpx SOURCES src/kokkos_hpx.cpp)
target_link_libraries(kokkos_hpx PRIVATE Kokkos::kokkos kokkos_hpx_interop)
