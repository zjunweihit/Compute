#include <iostream>
#include <iterator>
#include <boost/compute.hpp>
#include <boost/test/minimal.hpp>
#include <CL/sycl.hpp>

using namespace cl::sycl;

// Size of the Vector
constexpr size_t N = 8;

#define VECTOR 1
#ifdef VECTOR
using Vector = float[N];
#endif

int test_main(int argc, char *argv[]) {
#ifdef VECTOR
    Vector A = { 1, 1, 1, 1, 1, 1, 1, 1 };
    Vector B = { 1, 2, 3, 4, 5, 6, 7, 8 };
    Vector C;
#else
    std::vector<float> A;
    std::vector<float> B (N, 1);

    for (int i = 0; i < N; i++)
        A.push_back(i + 1.0);
#endif

    // Create a queue to work on default OpenCL queue
    queue q { boost::compute::system::default_queue() };

    buffer<float> a { std::begin(A), std::end(A) };
    buffer<float> b { std::begin(B), std::end(B) };
#ifndef VECTOR
    buffer<float> c { N };
#endif
    {
#ifdef VECTOR
        // NOTE: buffer c of N float using the storage of C!!
        // we can access C later directly with expected results.
        buffer<float> c { C, N };
#endif

        // Create a program from source
        auto program = boost::compute::program::create_with_source(R"(
            __kernel void vector_add(const __global float *a,
                                     const __global float *b,
                                     __global float *c)
            {
                c[get_global_id(0)] = a[get_global_id(0)] + b[get_global_id(0)];
            }
        )", boost::compute::system::default_context());

        // Build the program
        program.build();

        // Get the kernel from the program
        kernel k { boost::compute::kernel { program, "vector_add" } };

        // Launch the kernel: c = a + b
        q.submit([&] (handler &cgh) {
            cgh.set_arg(0, a.get_access<access::mode::read>(cgh));
            cgh.set_arg(1, b.get_access<access::mode::read>(cgh));
            cgh.set_arg(2, c.get_access<access::mode::write>(cgh));
            cgh.parallel_for(N, k);
        });
    } // End scope, so wait for the queue to complete.

    std::cout << std::endl << "A:" << std::endl;
    for (auto e : A)
        std::cout << e << ' ';
    std::cout << "\n";

    std::cout << std::endl << "B:" << std::endl;
    for (auto e : B)
        std::cout << e << ' ';
    std::cout << "\n";

#ifndef VECTOR
    auto C = c.get_access<access::mode::read>();
#endif
    std::cout << std::endl << "C = A + B:" << std::endl;
    for (auto e : C)
        std::cout << e << ' ';
    std::cout << "\n";

    std::cout << "Accurate computation!" << std::endl;
    return 0;
}
