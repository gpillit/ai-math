/* GEMV int8 x int8 -> int32 -> float, per l'lm_head (embedding tied). */
#include "aim_model.h"
#include <stdlib.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

typedef struct {
    const int8_t *W; const float *ws; int rows, cols; const int16_t *xq; float *y; float sx;
    atomic_int next;
} i8_task;
static void i8_task_fn(void *ctx, int tid, int nth);

void aim_gemv_i8(const int8_t *W, const float *ws, int rows, int cols,
                 const float *x, float *y)
{
    float amax = 0;
    for (int c = 0; c < cols; c++) { float a = fabsf(x[c]); if (a > amax) amax = a; }
    const float sx = amax > 0 ? amax / 127.0f : 1.0f, inv = 1.0f / sx;
    int16_t *xq = malloc((size_t)cols * sizeof(int16_t));
    for (int c = 0; c < cols; c++) { float v = x[c] * inv; xq[c] = (int16_t)(v >= 0 ? v + 0.5f : v - 0.5f); }

    i8_task k = { W, ws, rows, cols, xq, y, sx, 0 };
    aim_pool_run(i8_task_fn, &k);
    free(xq);
}

static void i8_task_fn(void *ctx, int tid, int nth)
{
    (void)tid; (void)nth;
    i8_task *k = ctx;
    const int cols = k->cols, rows = k->rows;
    const int8_t *W = k->W; const int16_t *xq = k->xq; const float sx = k->sx;
    for (int r0; (r0 = aim_pool_next(&k->next, 512)) < rows;)
    for (int r = r0; r < r0 + 512 && r < rows; r++) {
        const int8_t *w = W + (size_t)r * cols;
        int32_t sum;
#if defined(__AVX2__)
        __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
        int c = 0;
        for (; c + 32 <= cols; c += 32) {
            __m256i w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(w + c)));
            __m256i w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(w + c + 16)));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(w0, _mm256_loadu_si256((const __m256i *)(xq + c))));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(w1, _mm256_loadu_si256((const __m256i *)(xq + c + 16))));
        }
        a0 = _mm256_add_epi32(a0, a1);
        __m128i s = _mm_add_epi32(_mm256_castsi256_si128(a0), _mm256_extracti128_si256(a0, 1));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
        sum = _mm_cvtsi128_si32(s);
        for (; c < cols; c++) sum += w[c] * xq[c];
#else
        sum = 0;
        for (int c = 0; c < cols; c++) sum += w[c] * xq[c];
#endif
        k->y[r] = k->ws[r] * sx * (float)sum;
    }
}
