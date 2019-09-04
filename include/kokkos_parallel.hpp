///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2019 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

/// \file
/// Contains parallel algorithms overloads for Kokkos executors. Having
/// the overloads avoids the extra step of going through the executor interface.

#ifndef HPX_COMPUTE_KOKKOS_PARALLEL_HPP
#define HPX_COMPUTE_KOKKOS_PARALLEL_HPP

#include <kokkos_executors.hpp>

#include <hpx/include/iostreams.hpp>
#include <hpx/include/compute.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/executors/execution.hpp>
#include <hpx/parallel/executors/static_chunk_size.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

// TODO: This should most likely be in HPX proper.
namespace hpx { namespace parallel { namespace execution {
template <typename Policy>
struct is_task_policy : std::false_type {};

template <>
struct is_task_policy<parallel::execution::sequenced_task_policy> : std::true_type {};

template <>
struct is_task_policy<parallel::execution::parallel_task_policy> : std::true_type {};
}}}

namespace hpx {
namespace parallel {
template <
    typename Executor, typename Parameters, typename I, typename F,
    HPX_CONCEPT_REQUIRES_((!hpx::traits::is_iterator<I>::value &&
                           std::is_integral<I>::value) &&
                          compute::kokkos::is_kokkos_executor<Executor>::value)>
void for_loop(execution::parallel_policy_shim<Executor, Parameters> exec,
              typename std::decay<I>::type first, I last, F &&f) {
  static_assert(std::is_integral<I>::value,
                "for_loop overload for Kokkos executor can only be used with "
                "integral ranges. Use HPX executors or rewrite your range to "
                "use "
                "integral types.");

  std::cout << "customized for_loop for kokkos_executor" << std::endl;

  hpx::compute::kokkos::parallel_for(
      hpx::compute::kokkos::RangePolicy<typename Executor::execution_space>(
          first, last),
      KOKKOS_LAMBDA(I i) {
        printf("calling user provided function with argument i = %d\n", i);
        f(i);
      });

  typename Executor::execution_space().fence();
}

template <
    typename Parameters, typename I, typename F,
    HPX_CONCEPT_REQUIRES_((!hpx::traits::is_iterator<I>::value &&
                           std::is_integral<I>::value))>
hpx::future<void> for_loop(execution::parallel_task_policy_shim<compute::kokkos::kokkos_executor<Kokkos::Cuda>, Parameters> exec,
              typename std::decay<I>::type first, I last, F &&f) {
  static_assert(std::is_integral<I>::value,
                "for_loop overload for Kokkos executor can only be used with "
                "integral ranges. Use HPX executors or rewrite your range to "
                "use "
                "integral types.");

  std::cout << "customized for_loop for kokkos_executor with task_policy" << std::endl;

  hpx::compute::kokkos::parallel_for(
      hpx::compute::kokkos::RangePolicy<Kokkos::Cuda>(
          first, last),
      KOKKOS_LAMBDA(I i) {
        printf("calling user provided function with argument i = %d\n", i);
        f(i);
      });

  hpx::compute::cuda::target t{};
  return t.get_future();
}

// template <
//     typename Executor, typename Parameters, typename FwdIter, typename T,
//     typename F,
//     HPX_CONCEPT_REQUIRES_((hpx::traits::is_iterator<FwdIter>::value) &&
//                           compute::kokkos::is_kokkos_executor<Executor>::value)>
// inline typename util::detail::algorithm_result<
//     hpx::parallel::execution::parallel_policy_shim<Executor, Parameters>,
//     T>::type
// reduce(
//     hpx::parallel::execution::parallel_policy_shim<Executor, Parameters>
//     policy, FwdIter first, FwdIter last, T init, F &&f) {
//   // static_assert(
//   //     std::is_integral<decltype(*first)>::value &&
//   //         std::is_integral<decltype(*last)>::value,
//   //     "reduce overload for Kokkos executor can only be used with "
//   //     "integral ranges. Use HPX executors or rewrite your range to use "
//   //     "integral types.");

//   std::cout << "customized reduce for kokkos_executor" << std::endl;

//   // HPX_ASSERT_MSG(false, "this doesn't work!");

//   std::size_t n = std::distance(first, last);

//   hpx::compute::kokkos::parallel_reduce(
//       hpx::compute::kokkos::RangePolicy<typename
//       Executor::execution_space>(0,
//                                                                             n),
//       KOKKOS_LAMBDA(typename hpx::compute::kokkos::RangePolicy<
//                         typename Executor::execution_space>::member_type i,
//                     T & update){
//           // NOTE: This can't really be done. Or can it?
//           // Map a reduction function (T, T) -> T and an iterator, to (index,
//           // T&) and an integer range.
//           // update = f(first[i], update);
//           update = f(first[i], 0);
//       },
//       init);

//   hpx::compute::kokkos::fence();

//   return init;
// }

} // namespace parallel
} // namespace hpx

#endif
