//  Copyright (c) 2020 ETH Zurich
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// Contains helper functions for getting iterators from Kokkos views.
/// Currently simply returns View::data() for begin, and View::data() +
/// View::size() for end.

#pragma once

#include <Kokkos_Core.hpp>

namespace hpx {
namespace kokkos {
// These two functions do no checking of the validity of using v.data()
// directly as an iterator. They do not take strides, ranks, etc. into account.
template <typename V> auto view_begin(V const &v) { return v.data(); }

template <typename V> auto view_end(V const &v) { return v.data() + v.size(); }
} // namespace kokkos
} // namespace hpx
