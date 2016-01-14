#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
#include <algorithm>

#include <lift/memory.h>
#include <lift/backends.h>
#include <lift/parallel.h>
#include <lift/atomics.h>

#include <lift/sys/cuda/compute_device_cuda.h>
#include <lift/sys/host/compute_device_host.h>
using namespace lift;

template <target_system system>
struct vector_operations
{
    template<typename d_type>
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

    template<typename d_type>
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

    template<typename d_type>
    struct subtract 
    {
        pointer<system, d_type> data;
        d_type amount;

        subtract(pointer<system, d_type> data, d_type amount)
            : data(data), amount(amount)
        { }

        LIFT_HOST_DEVICE void operator() (const int index)
        {
           data[index] = data[index] -  amount;
        }
    };

    template<typename d_type>
    struct multiply
    {
        pointer<system, d_type> data;
        d_type val;

        multiply(pointer<system, d_type> data, d_type val)
            : data(data), val(val) 
        { }

        LIFT_HOST_DEVICE void operator() (const int index)
        {
            data[index] = data[index] * val;
        }
    };

    template<typename d_type>
    struct divide
    {
        pointer<system, d_type> data;
        d_type denominator;

        divide(pointer<system, d_type> data, d_type denominator)
            : data(data), denominator(denominator) 
        { }

        LIFT_HOST_DEVICE void operator() (const int index)
        {
            data[index] = data[index] / denominator;
        }
    };

    template<typename d_type>
    struct sum
    {
        pointer<system, d_type> data;
        d_type &sum_val;
        sum(pointer<system, d_type> data, d_type &sum_val)
           : data(data), sum_val(sum_val) 
        { } 
        LIFT_HOST_DEVICE void operator() (const d_type val)
        {
            sum_val += val;
        }
    };

    template<typename d_type>
    struct fill
    {
        pointer<system, d_type> data;
        d_type value;

        fill(pointer<system, d_type> &data, d_type value) 
            : data(data), value(value)
        { } 

        LIFT_HOST_DEVICE void operator() (const int index)
        {
            data[index] = value;
        }
    };

    template<typename d_type>
    static void fill_n(pointer<system, d_type> &data, d_type value)
    {
        parallel<system>::for_each(data.size(), fill<d_type>(data, value));
    }

    template<typename d_type>
    static void fill_n(pointer<system, d_type> &data, d_type value, int2 range)
    {
        parallel<system>::for_each(range, fill<d_type>(data, value));
    }

    template<typename d_type>
    static void fill_n(pointer<system, d_type> &data, d_type value, int end)
    {
        parallel<system>::for_each(end, fill<d_type>(data, value));
    }
};

typedef enum 
{
    RANGE,
    END,
} test_type;

template <target_system system>
struct tests
{
    template <typename d_type>
    static int for_each_test(pointer<system, d_type> data)
    { return 0; }
};

template<>
template<typename d_type>
int tests<cuda>::for_each_test(pointer<cuda, d_type> data)
{
    allocation<cuda, d_type> sum(1);
    vector_operations<cuda>::fill_n<d_type>(sum, 0);

    //Testing Pointer Variant of For Each
    parallel<cuda>::for_each(data, 
                      vector_operations<cuda>::p_add<d_type>(data, 10));
    //Testing Pointer Iteration Variant of For Each
    parallel<cuda>::for_each(data.begin(), data.end(), 
                      vector_operations<cuda>::p_add<d_type>(data, 10));
    //Testing Range Variant of For Each
    parallel<cuda>::for_each({data.size()/2, data.size()}, 
                      vector_operations<cuda>::add<d_type>(data, 10));
    //Testing End Variant of For Each
    parallel<cuda>::for_each(data.size(), 
                      vector_operations<cuda>::subtract<d_type>(data, 5));

    allocation<host, d_type> h_sum;
    h_sum.copy(data);
     for (int i = 0; i < data.size(); i++)
    {
        printf("%d ", h_sum[i]);
    }
    printf("gpu_sum: %d\n", h_sum[0]); 
    return 0;
}

template<>
template<typename d_type>
int tests<host>::for_each_test(pointer<host, d_type> data)
{
    d_type sum = 0;

    //Testing Pointer Variant of For Each
    parallel<host>::for_each(data, 
                       vector_operations<host>::p_add<d_type>(data, 10));
    //Testing Pointer Iteration Variant of For Each
    parallel<cuda>::for_each(data.begin(), data.end(), 
                      vector_operations<cuda>::p_add<d_type>(data, 10));
     //Testing Range Variant of For Each
    parallel<host>::for_each({data.size()/2, data.size()}, 
                      vector_operations<host>::add<d_type>(data, 10));
    //Testing End Variant of For Each
    parallel<host>::for_each(data.size(), 
                      vector_operations<host>::subtract<d_type>(data, 5));
   
    return sum;
}

bool for_each_unit_test(bool test_gpu)
{ 
    /*
      With the foreach tests we need to validate that:
      1. The operation is done using each value in the array
      2. The operation that has been done is the operation we expect
      3. Equivalent operations should produce equivalent 
         results on the cpu and gpu
    */
     if (test_gpu)
     {
         allocation<cuda, int> gpu_alloc(1000);
         vector_operations<cuda>::fill_n<int>(gpu_alloc, 0);
         tests<cuda>::for_each_test<int>(gpu_alloc);
     }
     allocation<host, int> cpu_alloc(1000);
     vector_operations<host>::fill_n<int>(cpu_alloc, 0);
     tests<host>::for_each_test<int>(cpu_alloc);

     return true;
}

int main(int argc, char **argv)
{        
    cuda_device_config gpu_requirements;
    std::vector<cuda_device_config> gpus;
    std::string error;

    cuda_device_config::enumerate_gpus(gpus, error, gpu_requirements);

    std::string cpu_runtime_version;
    compute_device *host_device = new compute_device_host();

    if (gpus.size() > 0)
    {
        compute_device *gpu_device  = new compute_device_cuda(gpus[0]);
        for_each_unit_test(true);
    }
    else
    {
        for_each_unit_test(false);
    }
    return 0;
}
