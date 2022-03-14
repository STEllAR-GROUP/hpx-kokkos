# Copyright (c) 2020-2022 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

export CRAYPE_LINK_TYPE=dynamic
export APPS_ROOT="/apps/daint/SSL/HPX/packages"
export CXX_STD="17"
export HWLOC_ROOT="${APPS_ROOT}/hwloc-2.0.3-gcc-8.3.0"
export BOOST_ROOT="${APPS_ROOT}/boost-1.75.0-gcc-10.1.0-c++17-debug/"

module load daint-gpu
module load cudatoolkit/11.0.2_3.38-8.1__g5b73779
spack load cmake@3.18.6
spack load ninja@1.10.0

export CXX=`which CC`
export CC=`which cc`

configure_extra_options="-DCMAKE_BUILD_TYPE=${build_type}"
configure_extra_options+=" -DHPX_KOKKOS_ENABLE_TESTS=ON"
configure_extra_options+=" -DHPX_KOKKOS_CUDA_FUTURE_TYPE=${future_type}"
configure_extra_options+=" -DKokkos_LAUNCH_COMPILER=OFF"

hpx_configure_extra_options="-DCMAKE_BUILD_TYPE=Debug"
hpx_configure_extra_options+=" -DHPX_WITH_EXAMPLES=OFF"
hpx_configure_extra_options+=" -DHPX_WITH_UNITY_BUILD=ON"
hpx_configure_extra_options+=" -DHPX_WITH_MALLOC=system"
hpx_configure_extra_options+=" -DHPX_WITH_NETWORKING=OFF"
hpx_configure_extra_options+=" -DHPX_WITH_FETCH_ASIO=ON"
hpx_configure_extra_options+=" -DHPX_WITH_CXX${CXX_STD}=ON"
hpx_configure_extra_options+=" -DHPX_WITH_CUDA=ON"
hpx_configure_extra_options+=" -DCMAKE_CUDA_COMPILER=$(which $CXX)"
hpx_configure_extra_options+=" -DCMAKE_CUDA_ARCHITECTURES=60"

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

# This is a workaround for a bug in the Cray clang compiler and/or Boost. When
# compiling in device mode, Boost detects that float128 is available, but
# compilation later fails with an error message saying float128 is not
# available for the target.
#
# This sets a custom Boost user configuration, which is a concatenation of the
# clang and nvcc compiler configurations, with the exception that
# BOOST_HAS_FLOAT128 is unconditionally disabled.
export CXXFLAGS="-I/dev/shm/hpx/src/.jenkins/cscs/ -DBOOST_USER_CONFIG='<boost_user_config_cray_clang.hpp>'"
