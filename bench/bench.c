/* Benchmark: quanto costa un GEMV "un token" su una matrice tipica di LLM,
 * misurato in banda effettiva verso la RAM. Il numero che conta per
 * l'inferenza su CPU è  (byte letti per token) / (GB/s disponibili). */
#include "aim.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*kernel_fn)(void *ctx, const float *x, float *y);

static double bench(kernel_fn f, void *ctx, const float *x, float *y, int iters)
{
    f(ctx, x, y);                       /* warm-up */
    double best = 1e30;
    for (int i = 0; i < iters; i++) {
        double t0 = aim_now_sec();
        f(ctx, x, y);
        double t = aim_now_sec() - t0;
        if (t < best) best = t;
    }
    return best;
}

typedef struct { const float *W; int rows, cols; } f32ctx;
static void k_f32(void *c, const float *x, float *y) { f32ctx *m = c; aim_gemv_f32(m->W, x, y, m->rows, m->cols); }
static void k_t3ref(void *c, const float *x, float *y) { aim_t3_gemv_ref(c, x, y); }
static void k_t3lut(void *c, const float *x, float *y) { aim_t3_gemv_lut(c, x, y); }
static void k_t3blk(void *c, const float *x, float *y) { aim_t3_gemv_lut_blocked(c, x, y); }
static void k_t3fac(void *c, const float *x, float *y) { aim_t3_gemv_lut_factored(c, x, y); }
static void k_t3simd(void *c, const float *x, float *y) { aim_t3_gemv_simd(c, x, y); }

int main(int argc, char **argv)
{
    aim_pool_init(0);
    int rows = argc > 1 ? atoi(argv[1]) : 4096;
    int cols = argc > 2 ? atoi(argv[2]) : 4096;
    int iters = argc > 3 ? atoi(argv[3]) : 20;
    printf("GEMV %d x %d, %d iterazioni (best-of)\n\n", rows, cols, iters);

    float *W = malloc((size_t)rows * cols * sizeof(float));
    float *x = malloc(cols * sizeof(float));
    float *y_ref = malloc(rows * sizeof(float)), *y = malloc(rows * sizeof(float));
    aim_fill_gaussian(W, (size_t)rows * cols, 1);
    aim_fill_gaussian(x, cols, 2);

    aim_t3_mat m;
    aim_t3_quantize(W, rows, cols, &m);

    size_t bytes_f32 = (size_t)rows * cols * 4, bytes_t3 = aim_t3_bytes(&m);
    printf("%-22s %12s %10s %12s %10s\n", "kernel", "bytes pesi", "ms", "GB/s eff.", "err vs f32");

    f32ctx fc = { W, rows, cols };
    double t = bench(k_f32, &fc, x, y_ref, iters);
    printf("%-22s %12zu %10.3f %12.1f %10s\n", "fp32", bytes_f32, t * 1e3, bytes_f32 / t / 1e9, "0");

    t = bench(k_t3ref, &m, x, y, iters);
    double e = aim_rel_err(y_ref, y, rows);
    printf("%-22s %12zu %10.3f %12.1f %10.3f\n", "ternary b3 ref", bytes_t3, t * 1e3, bytes_t3 / t / 1e9, e);

    t = bench(k_t3lut, &m, x, y, iters);
    e = aim_rel_err(y_ref, y, rows);
    printf("%-22s %12zu %10.3f %12.1f %10.3f\n", "ternary b3 LUT", bytes_t3, t * 1e3, bytes_t3 / t / 1e9, e);

    t = bench(k_t3blk, &m, x, y, iters);
    e = aim_rel_err(y_ref, y, rows);
    printf("%-22s %12zu %10.3f %12.1f %10.3f\n", "ternary b3 LUT blocked", bytes_t3, t * 1e3, bytes_t3 / t / 1e9, e);

    t = bench(k_t3fac, &m, x, y, iters);
    e = aim_rel_err(y_ref, y, rows);
    printf("%-22s %12zu %10.3f %12.1f %10.3f\n", "ternary b3 LUT 27x9", bytes_t3, t * 1e3, bytes_t3 / t / 1e9, e);

    aim_t3_tiled tl;
    aim_t3_tile(&m, &tl);
    t = bench(k_t3simd, &tl, x, y, iters);
    e = aim_rel_err(y_ref, y, rows);
    printf("%-22s %12zu %10.3f %12.1f %10.3f\n", "ternary b3 27x9 AVX2", bytes_t3, t * 1e3, bytes_t3 / t / 1e9, e);
    for (int B = 8; B <= 64; B *= 4) {
        float *X = malloc((size_t)B * cols * sizeof(float)), *Y = malloc((size_t)B * rows * sizeof(float));
        aim_fill_gaussian(X, (size_t)B * cols, 9);
        aim_t3_gemm_simd(&tl, X, cols, B, Y, rows);
        double best = 1e30;
        for (int i = 0; i < iters; i++) {
            double t0 = aim_now_sec(); aim_t3_gemm_simd(&tl, X, cols, B, Y, rows); double tt = aim_now_sec() - t0;
            if (tt < best) best = tt;
        }
        char name[32]; snprintf(name, sizeof name, "gemm VNNI B=%d", B);
        printf("%-22s %12zu %10.3f %12.1f %10s  (%.1f ms/token, %.0f GOPS)\n", name, bytes_t3, best * 1e3,
               bytes_t3 / best / 1e9, "-", best * 1e3 / B, 2.0 * rows * cols * B / best / 1e9);
        free(X); free(Y);
    }
    aim_t3_tiled_free(&tl);

    printf("\ncompressione pesi: %.1fx   (%.2f bit/peso incluse le scale)\n",
           (double)bytes_f32 / bytes_t3, 8.0 * bytes_t3 / ((double)rows * cols));
    printf("nota: 'err vs f32' e' l'errore di quantizzazione su pesi gaussiani casuali,\n"
           "      non l'errore del kernel (che e' ~0, vedi tests).\n");

    aim_t3_free(&m);
    if (W) free(W);
    free(x); free(y_ref); free(y);
    return 0;
}
