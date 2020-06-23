get_filename_component(HPXKokkos_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
include(CMakeFindDependencyMacro)

find_dependency(HPX 1.5.0 REQUIRED)
find_dependency(Kokkos REQUIRED)

include("${HPXKokkos_CMAKE_DIR}/HPXKokkosTargets.cmake")
