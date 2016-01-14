Lift for Thrust users
=====================

For users familiar with NVIDIA's Thrust, it is fairly easy to define Lift based on where it differs from Thrust.

Lift offers similar functionality to Thrust, but with important design distinctions.


POD Memory Containers
---------------------

Thrust's ``device_vector`` containers are non-POD types. They implement ``std::vector`` semantics, which means that the copy constructor and assignment operators create a new allocation and perform a memory copy, and the destructor releases the memory. While this approach makes sense in terms of preserving the interfaces programmers are already familiar with, it forces GPU code to be written under semantics that were not meant for code that needs to work across different memory spaces: parameters can only be passed by value from CPU to GPU, which forces code to use ``thrust::device_vector`` containers on the host but raw pointers on the device.

Lift implements memory containers that expose *pointer semantics* instead. Lift's base memory container, the ``allocation`` class, is derived from Lift's tagged ``pointer`` class, which means that an allocation can be treated as a pointer.

In addition to this, all of Lift's memory containers are POD types [#pod_vtable]_, which enables them to be passed directly by value to GPU kernels. This allows code to use the same structures on both sides: CPU code tracks data using the same containers that are used on the GPU side to perform calculations.

There are drawbacks to this approach, however. Because Lift exposes pointer semantics and because it intends to allow pass-by-value for memory containers, object destructors are not allowed to release the memory automatically. This is because we can't keep track of references to the same memory, so it is up to the programmer to ensure that memory is released at the right time.



Tagged pointers
---------------

Similarly to Thrust, Lift implements the concept of a *tagged pointer*: every pointer in Lift is tagged with an enum value (``target_system``) that determines whether the pointer is a CPU or GPU pointer. Likewise, every primitive entrypoint in Lift is tagged the same way.

However, unlike Thrust, Lift implements strict pointer semantics. Whereas Thrust tagged pointers are smart enough to move data from GPU to CPU as needed for dereferencing, Lift does no such thing: dereferencing a GPU pointer on the CPU leads to undefined behavior. This makes the Lift tagged pointer implementation extremely simple and obvious, removing a lot of complexity from the library.

One consequence of this design decision is that slow operations can be more easily spotted in client code, as Lift forces you to express such operations explicitly. For instance, instead of dereferencing GPU pointers on the CPU, Lift forces you to call ``peek()``; instead of doing a PCI-E transfer implicitly via an assignment between memory containers, Lift forces you to call ``copy()``.


Explicit memory movement
------------------------

Because memory containers are POD, the assignment operator implements pointer semantics. This means that Lift does not perform implicit memory transfers across compute devices. Instead, it exposes explicit methods to perform such transfers. Because such transfers are slow, this has the side-effect of minimizing the amount of hard-to-find locations in the code that could perform a slow operation implicitly --- it follows Lift's design philosophy of making important steps in the flow of a program very obvious.


Compatibility with Thrust
-------------------------

Lift's goal is not to compete with Thrust, rather to provide an alternative implementation for certain bits of functionality that are more amenable to environments which Thrust doesn't target directly.

To that end, Lift's parallel primitives can be used with Thrust containers, as Lift can operate on generic iterators and not just Lift pointers. Conversely, Lift's memory containers were designed to be compatible with Thrust: besides the usual ``begin()`` and ``end()`` interface, Lift exposes ``t_begin()`` and ``t_end()``, which return Thrust tagged iterators. These can be used directly when calling Thrust parallel primitives and will cause Thrust to schedule the compute work on the right device.

Lift also exposes a ``backend_policy`` object which converts a Lift ``target_system`` backend into a Thrust execution policy. This can be used explicitly to notify Thrust of the target device when executing a parallel primitive.


.. rubric:: Footnotes

.. [#pod_vtable] This is not strictly true: some of the memory containers expose virtual methods, which imply a virtual function table needs to exist. In such cases, Lift takes care to ensure that *no virtual functions are exposed on the GPU interface*. This means that, while the CPU virtual function table pointer is copied by value to the GPU, it can not actually be used on the GPU, making this usage safe.
