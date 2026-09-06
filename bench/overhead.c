/* Costo fisso: regione OpenMP vuota e GEMV ternario minuscolo. */
#include "aim.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main(void)
{
    int N = 2000;
    volatile int sink = 0;
    double t0 = aim_now_sec();
    for (int i = 0; i < N; i++) {
        #pragma omp parallel
        { if (omp_get_thread_num() == 999) sink++; }
    }
    double t_par = (aim_now_sec() - t0) / N;
    t0 = aim_now_sec();
    for (int i = 0; i < N; i++) {
        #pragma omp parallel
        {
            #pragma omp for
            for (int g = 0; g < 512; g++) if (g == 99999) sink++;
            #pragma omp for schedule(dynamic, 2) nowait
            for (int g = 0; g < 40; g++) if (g == 99999) sink++;
        }
    }
    double t_par2 = (aim_now_sec() - t0) / N;

    float *W = malloc(32 * 2560 * sizeof(float)), x[2560], y[32];
    aim_fill_gaussian(W, 32 * 2560, 1); aim_fill_gaussian(x, 2560, 2);
    aim_t3_mat m; aim_t3_quantize(W, 32, 2560, &m);
    aim_t3_tiled t; aim_t3_tile(&m, &t);
    aim_t3_gemv_simd(&t, x, y);
    t0 = aim_now_sec();
    for (int i = 0; i < N; i++) aim_t3_gemv_simd(&t, x, y);
    double t_gemv = (aim_now_sec() - t0) / N;
    printf("thread %d: parallel vuota %.1f us | parallel+2 for %.1f us | gemv 32x2560 %.1f us\n",
           omp_get_max_threads(), t_par * 1e6, t_par2 * 1e6, t_gemv * 1e6);
    return 0;
}
