# Copyright (c) 2020-2022 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

export CRAYPE_LINK_TYPE=dynamic
export CXX_STD="14"

module load daint-gpu
module switch PrgEnv-cray PrgEnv-gnu
module load cudatoolkit/11.2.0_3.39-2.1__gf93aa1c
module load Boost/1.78.0-CrayGNU-21.09
module load hwloc/2.4.1
spack load cmake@3.18.6
spack load ninja@1.10.0

export CXX=`which CC`
export CC=`which cc`

configure_extra_options="-DCMAKE_BUILD_TYPE=${build_type}"
configure_extra_options+=" -DCMAKE_CXX_COMPILER=/dev/shm/kokkos/install/bin/nvcc_wrapper"
configure_extra_options+=" -DHPX_KOKKOS_ENABLE_TESTS=ON"
configure_extra_options+=" -DHPX_KOKKOS_CUDA_FUTURE_TYPE=${future_type}"

hpx_configure_extra_options="-DCMAKE_BUILD_TYPE=${build_type}"
hpx_configure_extra_options+=" -DHPX_WITH_EXAMPLES=OFF"
hpx_configure_extra_options+=" -DHPX_WITH_UNITY_BUILD=ON"
hpx_configure_extra_options+=" -DHPX_WITH_MALLOC=system"
hpx_configure_extra_options+=" -DHPX_WITH_NETWORKING=OFF"
hpx_configure_extra_options+=" -DHPX_WITH_DISTRIBUTED_RUNTIME=OFF"
hpx_configure_extra_options+=" -DHPX_WITH_FETCH_ASIO=ON"
hpx_configure_extra_options+=" -DHPX_WITH_CXX${CXX_STD}=ON"
hpx_configure_extra_options+=" -DHPX_WITH_CUDA=ON"
hpx_configure_extra_options+=" -DHWLOC_ROOT=${EBROOTHWLOC}"

kokkos_configure_extra_options="-DCMAKE_BUILD_TYPE=${build_type}"
kokkos_configure_extra_options+=" -DCMAKE_CXX_COMPILER=/dev/shm/kokkos/src/bin/nvcc_wrapper"
kokkos_configure_extra_options+=" -DKokkos_CXX_STANDARD=${CXX_STD}"
kokkos_configure_extra_options+=" -DKokkos_ENABLE_SERIAL=OFF"
kokkos_configure_extra_options+=" -DKokkos_ENABLE_HPX=ON"
kokkos_configure_extra_options+=" -DKokkos_ENABLE_CUDA=ON"
kokkos_configure_extra_options+=" -DKokkos_ENABLE_HPX_ASYNC_DISPATCH=ON"
kokkos_configure_extra_options+=" -DKokkos_ENABLE_CUDA_LAMBDA=ON"
kokkos_configure_extra_options+=" -DKokkos_ENABLE_CUDA_CONSTEXPR=ON"
kokkos_configure_extra_options+=" -DKokkos_ARCH_HSW=ON"
kokkos_configure_extra_options+=" -DKokkos_ARCH_PASCAL60=ON"
