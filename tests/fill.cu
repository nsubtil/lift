#include <lift/test/test.h>
#include <lift/test/check.h>

#include <lift/parallel.h>
#include <lift/backends.h>
#include <lift/memory/scoped_allocation.h>

using namespace lift;


//Unit test for vector-wide implementation of fill
template <target_system system>
void fill_vector()
{
    bool test_success = true;
    uint32 size = 1000;
    int fill_val = 10;

    allocation<system, int> data(size);
    parallel<system>::fill(data, fill_val);

    for (int i = 0; i < size; i++)
    {
        if (data.peek(i) != fill_val)
        {
            test_success = false;
        }
    }     
    LIFT_TEST_CHECK(test_success == true);
}
LIFT_TEST_FUNC(fill_vector_test, fill_vector);

template <target_system system>
void fill_input_iter()
{
    bool test_success = true;
    uint32 size = 1000;
    int fill_val = 10;

    allocation<system, int> data(size);
    parallel<system>::fill(data, fill_val);

    for (int i = 0; i < size; i++)
    {
        if (data.peek(i) != fill_val)
        {
            test_success = false;
        }
    }  
    LIFT_TEST_CHECK(test_success == true);
}
LIFT_TEST_FUNC(fill_input_iter_test, fill_vector);
