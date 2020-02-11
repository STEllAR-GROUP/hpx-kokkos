#include <hpx/hpx_main.hpp>
#include <hpx/include/util.hpp>
#include <hpx/kokkos.hpp>

#include <string>

struct kernel {
  Kokkos::View<double *> a;
  Kokkos::View<double *> b;
  Kokkos::View<double *> c;

  kernel(Kokkos::View<double *> a, Kokkos::View<double *> b,
         Kokkos::View<double *> c)
      : a(a), b(b), c(c) {}

  KOKKOS_INLINE_FUNCTION void operator()(std::size_t const i) const {
    for (std::size_t j = 0; j < 1000; j++) {
      a[i] += sqrt(pow(3.14159, i * j));
      b[i] += sin(sqrt(pow(3.14159, 7 * i * j)));
      c[i] += pow(a[i], b[i]);
    }
  }
};

struct scan_kernel {
  Kokkos::View<double *> a;

  scan_kernel(Kokkos::View<double *> a) : a(a) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, double &update, const bool final_pass) const {
    for (std::size_t j = 0; j < 1000; j++) {
      update += sin(pow(a(i), j * 13));
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(double &update) const { update = 0.0; }

  KOKKOS_INLINE_FUNCTION
  void join(double &update, const double &input) const {
    for (std::size_t j = 0; j < 1000; j++) {
      update += sin(pow(input, j * 13));
    }
  }
};

int main(int argc, char **argv) {
  {
    hpx::kokkos::ScopeGuard g(argc, argv);

    std::size_t n = 10000;
    if (argc == 2) {
      n = std::stoi(argv[1]);
    } else if (argc > 2) {
      std::cerr << "Usage: ./kokkos_hpx n" << std::endl;
      hpx::finalize();
      return 1;
    }

    std::cout << "Using n = " << n << std::endl;

    // Views are allocated on memory space of the default device (i.e. GPU if
    // available, otherwise host). These are blocking calls.
    std::cout << "constructing views... " << std::flush;
    hpx::kokkos::View<double *> a("a", n);
    hpx::kokkos::View<double *> b("b", n);
    hpx::kokkos::View<double *> c("c", n);
    hpx::kokkos::View<double *> d("d", n);
    hpx::kokkos::View<double *> e("e", n);
    hpx::kokkos::View<double *> f("f", n);
    Kokkos::fence();
    std::cout << "done" << std::endl;

    kernel k1(a, b, c);
    kernel k2(d, e, f);

    // Plain Kokkos parallel_for. This is asynchronous on most GPUs, but does
    // not return a future. The only way of waiting is to fence the whole
    // execution space which blocks the OS thread. This is *not* recommended.
    std::cout << "parallel_for... " << std::flush;
    hpx::kokkos::parallel_for(n, k1);
    hpx::kokkos::DefaultExecutionSpace().fence();
    std::cout << "done" << std::endl;

    // Kokkos parallel_for with a single execution space instance. The current
    // semantics are to serialize kernel launches on the same instance.
    // Different instances may have work launched concurrently, but must not.
    std::cout
        << "launching two asynchronous parallel_fors on the same instance... "
        << std::flush;
    auto inst = hpx::kokkos::make_execution_space<>();
    // Same as:
    // auto inst =
    // hpx::kokkos::make_execution_space_instance<hpx::kokkos::DefaultExecutionSpace>();
    // Can also get a host instance:
    // auto inst =
    // hpx::kokkos::make_execution_space_instance<hpx::kokkos::DefaultHostExecutionSpace>();
    // Or a specific instance:
    // auto inst =
    // hpx::kokkos::make_execution_space_instance<hpx::kokkos::Cuda>();
    hpx::util::high_resolution_timer t;
    hpx::future<void> f1 = hpx::kokkos::parallel_for_async(
        hpx::kokkos::RangePolicy<>(inst, 0, n), k1);

    hpx::future<void> f2 = hpx::kokkos::parallel_for_async(
        hpx::kokkos::RangePolicy<>(inst, 0, n), k2);

    std::cout << "returned... " << std::flush;
    hpx::wait_all(f1, f2);
    double tt = t.elapsed();
    std::cout << "done" << std::endl;
    std::cout << "serialized kernel launch took " << tt << " seconds"
              << std::endl;

    // Kokkos parallel_for with two different execution space instances. The
    // two kernel launches may be concurrent, but must not.
    std::cout << "launching two asynchronous parallel_fors on different "
                 "instances... "
              << std::flush;
    t.restart();

    hpx::future<void> f3 = hpx::kokkos::parallel_for_async(
        hpx::kokkos::RangePolicy<>(hpx::kokkos::make_execution_space(), 0, n),
        k1);

    hpx::future<void> f4 = hpx::kokkos::parallel_for_async(
        hpx::kokkos::RangePolicy<>(hpx::kokkos::make_execution_space(), 0, n),
        k2);

    std::cout << "returned... " << std::flush;
    hpx::wait_all(f3, f4);
    tt = t.elapsed();
    std::cout << "done" << std::endl;
    std::cout << "separated kernel launch took " << tt << " seconds"
              << std::endl;

    // Asynchronous parallel_reduce
    std::cout << "launching two asynchronous parallel_reduces on different "
                 "instances... "
              << std::flush;
    Kokkos::View<double> sum("sum");
    hpx::future<void> f5 = hpx::kokkos::parallel_reduce_async(
        hpx::kokkos::RangePolicy<>(hpx::kokkos::make_execution_space(), 0, n),
        KOKKOS_LAMBDA(int const &i, double &x) {
          for (std::size_t j = 0; j < 10000; ++j) {
            x += sqrt(pow(a(i), i * j / 7.0));
          }
        },
        sum);

    Kokkos::View<double> prod("prod");
    hpx::future<void> f6 = hpx::kokkos::parallel_reduce_async(
        hpx::kokkos::RangePolicy<>(hpx::kokkos::make_execution_space(), 0, n),
        KOKKOS_LAMBDA(int const &i, double &x) {
          for (std::size_t j = 0; j < 10000; ++j) {
            x *= sqrt(pow(d(i), i * j * 3));
          }
        },
        prod);

    hpx::wait_all(f5, f6);
    std::cout << "done" << std::endl;

    // All together
    hpx::future<void> f7 = hpx::kokkos::parallel_for_async(
        hpx::kokkos::RangePolicy<>(hpx::kokkos::make_execution_space(), 0, n),
        k1);
    hpx::future<void> f8 = hpx::kokkos::parallel_for_async(
        hpx::kokkos::RangePolicy<>(hpx::kokkos::make_execution_space(), 0, n),
        k2);

    // TODO: The parallel_scans don't seem to launch until the parallel_fors
    // have finished. Kokkos or device limitation?
    hpx::future<void> f9 = hpx::kokkos::parallel_scan_async(
        hpx::kokkos::RangePolicy<>(hpx::kokkos::make_execution_space(), 0, n),
        scan_kernel(f));
    hpx::future<void> f10 = hpx::kokkos::parallel_scan_async(
        hpx::kokkos::RangePolicy<>(hpx::kokkos::make_execution_space(), 0, n),
        scan_kernel(a));
    hpx::wait_all(f7, f8, f9, f10);
  }

  return hpx::finalize();
}
