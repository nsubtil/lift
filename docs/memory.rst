Memory Handling in Lift
=======================

This section describes the memory containers and memory-related abstractions available in Lift.


Lift Pointers
-------------

Lift's :lift:`pointer` class sits at the base of the hierarchy of memory abstractions.

A pointer is a class that implements pointer semantics, meant to look and act like a pointer. Lift pointers differ from raw pointers in two important ways:

* Lift pointers are *sized*: each pointer keeps track of the size of the memory region it points at
* Lift pointers are *tagged* with the memory space they belong to
* Lift pointers include the underlying data type in the type of the pointer itself

Pointer objects can be assigned to other pointer objects of compatible types. Lift will perform compile-time type checks to catch obvious mistakes upon assignment:

* The underlying value type for the pointers is checked at compile-time (see :lift:`check_memory_pointer_assignment_compatible`).
* The target system for both pointers is checked at compile-time. Cross-memory-space pointer assignment is allowed but is defined as generating a null pointer.


Cross-memory-space pointer assignment
"""""""""""""""""""""""""""""""""""""

The topic of cross-memory-space pointer assignment merits some explanation, since behavior differs between that and value-type checks.

When generating code that targets both the CPU and GPU, Lift requires the compiler to generate 4 versions of the code:

* ``target_system = host`` and CUDA decorator ``__host__``
* ``target_system = cuda`` and CUDA decorator ``__host__``
* ``target_system = host`` and CUDA decorator ``__device__``
* ``target_system = cuda`` and CUDA decorator ``__device__``

Two of these versions are not callable and are effectively pruned by later compilation stages, but they must be valid. This implies that implicit cross-memory-space pointer assignment will happen, so this must not be a compilation error.

The solution chosen for Lift is to allow cross-memory-space pointer assignment to compile, but force the LHS pointer in the assignment to change into a null pointer. Such errors should be easy to catch at runtime.


Lift Memory Containers
----------------------

Lift exposes a few different memory containers, implementing somewhat different behavior.


Allocation
""""""""""

The base allocation type in Lift is the aptly-named :lift:`allocation` class. This class derives from Lift's pointer class and is identical in all respects, except that it also implements some of the ``std::vector`` interface for handling memory allocations.

A Lift allocation can be thought of as a pointer that can reallocate itself. The reallocation policy is trivial: it always allocates exactly the amount of memory requested; when an allocation is shrunk, any memory that is no longer needed is released.

Typical uses for an allocation object are instances where a memory buffer is required and the buffer is not expected to change size often.

Note that because an allocation always holds the exact amount of memory required to store the data it holds, an interface similar to ``std::vector::push_back()`` would become very inefficient. For this reason, Lift's allocation class does not expose such an interface.


Persistent Allocation
"""""""""""""""""""""

The :lift:`persistent_allocation` class derives from allocation. It implements similar behavior, but it adds a distinction between capacity and size:

* The *size* is the amount of data that the persistent allocation holds
* The *capacity* is the amount of memory that was allocated to hold the data

*Capacity* can be bigger than *size*, which allows the persistent allocation to more efficiently accommodate reallocations, at the cost of using more memory than would be strictly required. The allocation policy is more involved:

* When a persistent allocation is shrunk, it will not make any changes to the underlying memory allocation
* When a persistent allocation is grown via a call to :lift:`persistent_allocation::resize`, the amount of memory allocated will match the size
* When :lift:`persistent_allocation::push_back` causes the memory allocation to grow, it is grown to double the size
* Calling :lift:`persistent_allocation::reserve` will always allocate exactly the amount requested

Use cases for persistent allocations are instances where a buffer is expected to hold variable amounts of data or instances where a buffer is populated by appending elements to the end.


Scoped Allocation
"""""""""""""""""

Lift's :lift:`scoped_allocation` class derives from persistent_allocation. It is identical in all respects, except that the destructor for scoped_allocation will free the underlying allocation.

Typical use is as a temporary buffer declared on the stack.
