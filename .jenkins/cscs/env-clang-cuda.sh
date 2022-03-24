# Copyright (c) 2020-2022 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

export CRAYPE_LINK_TYPE=dynamic
export APPS_ROOT="/apps/daint/SSL/HPX/packages"
export CXX_STD="17"
export HWLOC_ROOT="${APPS_ROOT}/hwloc-2.0.3-gcc-8.3.0"
export BOOST_ROOT="${APPS_ROOT}/boost-1.69.0-gcc-8.3.0-c++17-release/"

module load daint-gpu
module load cudatoolkit/11.0.2_3.38-8.1__g5b73779
module load CMake
spack load ninja@1.10.0

export CXX=`which CC`
export CC=`which cc`

configure_extra_options="-DCMAKE_BUILD_TYPE=${build_type}"
configure_extra_options+=" -DHPX_KOKKOS_ENABLE_TESTS=ON"
configure_extra_options+=" -DHPX_KOKKOS_CUDA_FUTURE_TYPE=${future_type}"
configure_extra_options+=" -DKokkos_LAUNCH_COMPILER=OFF"

hpx_configure_extra_options="-DCMAKE_BUILD_TYPE=${build_type}"
hpx_configure_extra_options+=" -DHPX_WITH_EXAMPLES=OFF"
hpx_configure_extra_options+=" -DHPX_WITH_UNITY_BUILD=ON"
hpx_configure_extra_options+=" -DHPX_WITH_MALLOC=system"
hpx_configure_extra_options+=" -DHPX_WITH_NETWORKING=OFF"
hpx_configure_extra_options+=" -DHPX_WITH_FETCH_ASIO=ON"
hpx_configure_extra_options+=" -DHPX_WITH_CXX${CXX_STD}=ON"
hpx_configure_extra_options+=" -DHPX_WITH_CUDA=ON"
hpx_configure_extra_options+=" -DHPX_WITH_CUDA_CLANG=ON"
hpx_configure_extra_options+=" -DHPX_CUDA_CLANG_FLAGS=\"--cuda-gpu-arch=sm_60\""

kokkos_configure_extra_options="-DCMAKE_BUILD_TYPE=${build_type}"
kokkos_configure_extra_options+=" -DKokkos_ENABLE_LAUNCH_COMPILER=OFF"
kokkos_configure_extra_options+=" -DKokkos_CXX_STANDARD=${CXX_STD}"
kokkos_configure_extra_options+=" -DKokkos_ENABLE_SERIAL=OFF"
kokkos_configure_extra_options+=" -DKokkos_ENABLE_HPX=ON"
kokkos_configure_extra_options+=" -DKokkos_ENABLE_CUDA=ON"
kokkos_configure_extra_options+=" -DKokkos_ENABLE_HPX_ASYNC_DISPATCH=ON"
kokkos_configure_extra_options+=" -DKokkos_ENABLE_CUDA_LAMBDA=ON"
kokkos_configure_extra_options+=" -DKokkos_ENABLE_CUDA_CONSTEXPR=ON"
kokkos_configure_extra_options+=" -DKokkos_ARCH_HSW=ON"
kokkos_configure_extra_options+=" -DKokkos_ARCH_PASCAL60=ON"
