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
