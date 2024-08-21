#pragma once

namespace LongServe {

#define CUDA_CHECK(cmd) do {                         \
    cudaError_t e = (cmd);                             \
    if (e != cudaSuccess) {                          \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n",      \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                          \
    }                                                \
} while(0)

/*
assert_whenever: assertion which ignore whether NDEBUG is set

In C++, assert() is evaluated only when NDEBUG is not set. This is
inconvenient when we want to check the assertion even in release mode.
This macro is a workaround for this problem.
*/

extern "C" {
// Copied from assert.h
extern void __assert_fail (const char *__assertion, const char *__file,
			   unsigned int __line, const char *__function)
     __THROW __attribute__ ((__noreturn__));

#define __ASSERT_FUNCTION	__extension__ __PRETTY_FUNCTION__
#  define assert_whenever(expr)							\
     (static_cast <bool> (expr)						\
      ? void (0)							\
      : __assert_fail (#expr, __FILE__, __LINE__, __ASSERT_FUNCTION))
}

template<typename T>
T cdiv(const T &a, const T &b) {
    return (a + b - 1) / b;
}

#define INDEX_2D(dim0, dim1, pos0, pos1) ((pos0) * (dim1) + (pos1))
#define INDEX_3D(dim0, dim1, dim2, pos0, pos1, pos2) ((pos0) * (dim1) * (dim2) + (pos1) * (dim2) + (pos2))
#define INDEX_4D(dim0, dim1, dim2, dim3, pos0, pos1, pos2, pos3) ((pos0) * (dim1) * (dim2) * (dim3) + (pos1) * (dim2) * (dim3) + (pos2) * (dim3) + (pos3))

}

