#include <lift/sys/host/compute_device_host.h>

#if __x86_64__
#include <lift/sys/host/x86_64/vector_flags.h>
#else
#error "unsupported architecture"
#endif // __x86_64__

const char *cache_type_strings[] = {
    "(null)",
    "data",
    "instruction",
    "unified"
};

int main(int argc, char **argv)
{
    lift::compute_device_host cpu;

    printf("%s\n", cpu.get_name());
    printf("vector extensions:");

#if __x86_64__
#define VEC(ext) \
    if (cpu.config.vector_extensions & lift::x86_64::ext) \
        printf(" %s", "" #ext);

    VEC(SSE);
    VEC(SSE2);
    VEC(SSE3);
    VEC(SSE3_S);
    VEC(SSE4_1);
    VEC(SSE4_2);
    VEC(SSE4_a);
    VEC(SSE_XOP);
    VEC(SSE_FMA4);
    VEC(SSE_FMA3);
    VEC(AVX);
    VEC(AVX2);

#undef VEC
#endif // __x86_64__


    if (cpu.config.caches.size())
    {
        printf("\n\n");
        printf("cache topology:\n");

        for(auto cache : cpu.config.caches)
        {
            printf(" L%d: %s %d KB %u-way %u bytes/line\n",
                   cache.level,
                   cache_type_strings[cache.type],
                   cache.total_size / 1024,
                   cache.associativity,
                   cache.line_size);
        }
    }

    return 0;
}
