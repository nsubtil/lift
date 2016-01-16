Introduction to Lift
====================

Lift is a library of components useful for parallel programming. It exposes capabilities similar to NVIDIA's Thrust, but using somewhat different semantics aimed at creating a library that is more robust, transparent and suitable for production code.


Design Goals
------------

The goal of Lift is to provide implementations for common building blocks when writing compute code for GPU and many-core CPU machines:

* Device-specific memory containers
* Cross-device memory transfers
* Well-known parallel primitives
* Interoperability with existing libraries in the same space

The general design philosophy can be summarized:

* Expose *slow* operations explicitly
* Abstract common operations where this can be done with minimal performance impact
* Provide a code base that is easily understood
* Avoid hidden functionality
* Provide a stable, predictable code base


Quick Example
-------------

Sort a set of key-value pairs::

    #include <lift/memory.h>
    #include <lift/backends.h>
    #include <lift/parallel.h>

    using namespace lift;

    template <target_system system>
    void test(void)
    {
        scoped_allocation<system, int> buf = { 3, 2, 5, 8, 10, 29, 4, 76, 6 };
        scoped_allocation<system, int> temp_keys(buf.size());
        scoped_allocation<system, uint8> temp_storage;

        parallel<system>::sort(buf, temp_keys, temp_storage);

        printf("%d %d %d\n", buf.peek(0), buf.peek(1), buf.peek(2));
    }

    int main(int argc, char **argv)
    {
        test<host>();
        test<cuda>();
    }
