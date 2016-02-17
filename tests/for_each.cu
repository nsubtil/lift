#include <lift/test/test.h>
#include <lift/test/check.h>

#include <lift/memory.h>
#include <lift/backends.h>
#include <lift/parallel.h>
#include <lift/atomics.h>

using namespace lift;

const uint32 vec_len = 10000;

template<target_system system, typename d_type>
struct p_add
{
    pointer<system, d_type> data;
    d_type amount;
    p_add(pointer<system, d_type> data, d_type amount) 
        : data(data), amount(amount)
    { }

    LIFT_HOST_DEVICE void operator() (d_type &ind)
    {
        ind += amount;
    }
};

template<target_system system, typename d_type>
struct add 
{
    pointer<system, d_type> data;
    d_type amount;

    add(pointer<system, d_type> data, d_type amount)
        : data(data), amount(amount)
    { }

    LIFT_HOST_DEVICE void operator() (const int index)
    {
       data[index] += amount;
    }
};

template <target_system system>
void for_each_end()
{
    int sub_num = 10; 
    allocation<system, int> data(vec_len);
    parallel<system>::fill(data, 0);
    parallel<system>::for_each(data.size(), add<system, int>(data, sub_num));

    for (int i = 0; i < data.size(); i++)
    {
        LIFT_TEST_CHECK(data.peek(i) == sub_num)
    }
}
LIFT_TEST_FUNC(for_each_end_test, for_each_end);

template <target_system system>
void for_each_iter()
{
    int add_num = 10; 
    allocation<system, int> data(vec_len);
    parallel<system>::fill(data, 0);
    parallel<system>::for_each(data.begin(), data.end(), 
                               p_add<system, int>(data, add_num));

    for (int i = 0; i < data.size(); i++)
    {
        LIFT_TEST_CHECK(data.peek(i) == add_num)
    }
}
LIFT_TEST_FUNC(for_each_iter_test, for_each_iter);

template <target_system system>
void for_each_pointer()
{
    int add_num = 10; 
    allocation<system, int> data(vec_len);
    parallel<system>::fill(data, 0);
    parallel<system>::for_each(data, p_add<system, int>(data, add_num));

    for (int i = 0; i < data.size(); i++)
    {
        LIFT_TEST_CHECK(data.peek(i) == add_num)
    }
}
LIFT_TEST_FUNC(for_each_pointer_test, for_each_pointer);

template <target_system system>
void for_each_range()
{
    int add_num = 10; 
    allocation<system, int> data(vec_len);
    parallel<system>::fill(data, 0);
    parallel<system>::for_each({data.size()/2, data.size()},
                               add<system, int>(data, add_num));

    for (int i = 0; i < data.size()/2; i++)
    {
        LIFT_TEST_CHECK(data.peek(i) == 0)
    }
    for (int i = data.size()/2; i < data.size(); i++)
    {
        LIFT_TEST_CHECK(data.peek(i) == add_num)
    }
}
LIFT_TEST_FUNC(for_each_range_test, for_each_range);
