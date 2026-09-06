#include "aim.h"
#include <math.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

double aim_now_sec(void)
{
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/* xorshift64* + Box-Muller */
void aim_fill_gaussian(float *p, size_t n, uint64_t seed)
{
    uint64_t s = seed ? seed : 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < n; i += 2) {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        double u1 = ((s * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 9007199254740992.0);
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        double u2 = ((s * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 9007199254740992.0);
        if (u1 < 1e-300) u1 = 1e-300;
        double r = sqrt(-2.0 * log(u1));
        p[i] = (float)(r * cos(6.283185307179586 * u2));
        if (i + 1 < n) p[i + 1] = (float)(r * sin(6.283185307179586 * u2));
    }
}

double aim_rel_err(const float *ref, const float *got, int n)
{
    double num = 0, den = 0;
    for (int i = 0; i < n; i++) {
        double d = (double)ref[i] - got[i];
        num += d * d; den += (double)ref[i] * ref[i];
    }
    return den > 0 ? sqrt(num / den) : sqrt(num);
}
