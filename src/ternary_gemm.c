/* GEMM ternario per il prefill: B attivazioni sugli stessi pesi.
 *
 * La LUT del GEMV costa ~35 istruzioni per 32 byte di pesi *per token*: sul
 * prefill non si ammortizza. Qui ogni tile (32 righe) viene spacchettato una
 * volta sola in int8, con le colonne interleaved a 4 per riga, e moltiplicato
 * per tutti i B token con vpdpbusd (AVX-VNNI: 4 prodotti u8*s8 sommati in
 * int32 per lane) o con vpmaddubsw+vpmaddwd (AVX2).
 *
 * Le attivazioni int8 x_q vengono rese unsigned come x_q+128; la correzione
 * 128*sum_riga(w) si calcola con lo stesso kernel usando x = 128.
 *
 * Divisione per 3 su 16 bit: q = (b*171)>>9 e' esatta per b < 243. */
#include "aim.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__AVX2__)
#include <immintrin.h>

#if defined(__AVXVNNI__)
#define DPBUSD(acc, u, s) _mm256_dpbusd_avx_epi32(acc, u, s)
#else
static inline __m256i DPBUSD(__m256i acc, __m256i u, __m256i s)
{
    __m256i p = _mm256_maddubs_epi16(u, s);   /* |u*s| <= 255*2: nessuna saturazione */
    return _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_set1_epi16(1)));
}
#endif

typedef struct {
    const aim_t3_tiled *t;
    const uint8_t *xu;   /* [B][cols20]: x_q + 128 */
    const float *sx;     /* [B] */
    int B, cols20;
    float *Y; int ldy;
    atomic_int next;
} gemm_task;

/* Layout W4 di un tile: per blocco j di 4 colonne, 4 ymm (128 byte):
 *   ymm k, lane l (int32) = 4 byte s8 = w[row(k,l)][4j .. 4j+3]
 *   row(k,l) = l < 4 ? 4k + l : 16 + 4k + (l - 4)                        */
static inline int w4_row(int k, int l) { return l < 4 ? 4 * k + l : 16 + 4 * k + (l - 4); }

/* unpack generico: righe int8 dalla tabella del codice, poi layout W4 */
static void unpack_tile_generic(const aim_t3_tiled *t, int tile, int cols20, int8_t *W4, int8_t *rowbuf)
{
    memset(rowbuf, 0, (size_t)AIM_T3_TILE * cols20);
    int r0 = tile * AIM_T3_TILE, nr = t->rows_pad - r0 < AIM_T3_TILE ? t->rows_pad - r0 : AIM_T3_TILE;
    aim_code_unpack_rows(t, r0, nr, rowbuf, cols20);
    for (int j = 0; j < cols20 / 4; j++)
        for (int k = 0; k < 4; k++)
            for (int l = 0; l < 8; l++) {
                int r = w4_row(k, l);
                memcpy(W4 + (size_t)j * 128 + k * 32 + l * 4, rowbuf + (size_t)r * cols20 + 4 * j, 4);
            }
}

static void unpack_tile(const aim_t3_tiled *t, int tile, int cols20, int8_t *W4)
{
    const int G = t->G;
    const uint8_t *td = t->data + (size_t)tile * G * AIM_T3_TILE;
    const __m256i k171 = _mm256_set1_epi16(171), k3 = _mm256_set1_epi16(3), k1 = _mm256_set1_epi16(1);
    __m256i colv[20];
    for (int g0 = 0; g0 < cols20 / 5; g0 += 4) {            /* 4 gruppi = 20 colonne = 5 blocchi */
        for (int k = 0; k < 4; k++) {
            int g = g0 + k;
            __m256i w0, w1;
            if (g < G) {
                __m256i codes = _mm256_loadu_si256((const __m256i *)(td + (size_t)g * AIM_T3_TILE));
                w0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(codes));
                w1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(codes, 1));
            } else {
                w0 = w1 = _mm256_set1_epi16(121);             /* 121 = tutti i trit a 0 */
            }
            for (int d = 0; d < AIM_T3_GROUP; d++) {
                __m256i q0 = _mm256_srli_epi16(_mm256_mullo_epi16(w0, k171), 9);
                __m256i q1 = _mm256_srli_epi16(_mm256_mullo_epi16(w1, k171), 9);
                __m256i d0 = _mm256_sub_epi16(_mm256_sub_epi16(w0, _mm256_mullo_epi16(q0, k3)), k1);
                __m256i d1 = _mm256_sub_epi16(_mm256_sub_epi16(w1, _mm256_mullo_epi16(q1, k3)), k1);
                w0 = q0; w1 = q1;
                __m256i packed = _mm256_packs_epi16(d0, d1);          /* [d0.L, d1.L | d0.H, d1.H] */
                colv[k * 5 + d] = _mm256_permute4x64_epi64(packed, 0xD8);   /* righe 0..31 in ordine */
            }
        }
        for (int j = 0; j < 5; j++) {
            __m256i c0 = colv[4 * j], c1 = colv[4 * j + 1], c2 = colv[4 * j + 2], c3 = colv[4 * j + 3];
            __m256i p01l = _mm256_unpacklo_epi8(c0, c1), p01h = _mm256_unpackhi_epi8(c0, c1);
            __m256i p23l = _mm256_unpacklo_epi8(c2, c3), p23h = _mm256_unpackhi_epi8(c2, c3);
            int8_t *dst = W4 + ((size_t)g0 / 4 * 5 + j) * 128;
            _mm256_storeu_si256((__m256i *)(dst + 0),  _mm256_unpacklo_epi16(p01l, p23l));  /* righe 0-3  | 16-19 */
            _mm256_storeu_si256((__m256i *)(dst + 32), _mm256_unpackhi_epi16(p01l, p23l));  /* righe 4-7  | 20-23 */
            _mm256_storeu_si256((__m256i *)(dst + 64), _mm256_unpacklo_epi16(p01h, p23h));  /* righe 8-11 | 24-27 */
            _mm256_storeu_si256((__m256i *)(dst + 96), _mm256_unpackhi_epi16(p01h, p23h));  /* righe 12-15| 28-31 */
        }
    }
}

static inline void gemm_store(const gemm_task *k, int tile, int b, const __m256i *acc, const __m256i *corr)
{
    const aim_t3_tiled *t = k->t;
    const float sx = k->sx[b];
    float *y = k->Y + (size_t)b * k->ldy;
    int32_t tmp[8];
    for (int kk = 0; kk < 4; kk++) {
        _mm256_storeu_si256((__m256i *)tmp, _mm256_sub_epi32(acc[kk], corr[kk]));
        for (int l = 0; l < 8; l++) {
            int r = tile * AIM_T3_TILE + w4_row(kk, l);
            if (r < t->rows) y[r] = (float)tmp[l] * t->scale[r] * sx;
        }
    }
}

static void gemm_task_fn(void *ctx, int tid, int nth)
{
    (void)tid; (void)nth;
    gemm_task *k = ctx;
    const aim_t3_tiled *t = k->t;
    const int cols20 = k->cols20, nblk = cols20 / 4, ntiles = t->rows_pad / AIM_T3_TILE, B = k->B;
    int8_t *W4 = _mm_malloc((size_t)nblk * 128, 32);
    int8_t *rowbuf = t->code ? malloc((size_t)AIM_T3_TILE * cols20) : NULL;
    const __m256i x128 = _mm256_set1_epi8((char)128);

    for (int tile; (tile = aim_pool_next(&k->next, 1)) < ntiles;) {
        if (t->code) unpack_tile_generic(t, tile, cols20, W4, rowbuf);
        else unpack_tile(t, tile, cols20, W4);

        __m256i corr[4] = { _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256() };
        for (int j = 0; j < nblk; j++) {
            const int8_t *wp = W4 + (size_t)j * 128;
            for (int kk = 0; kk < 4; kk++)
                corr[kk] = DPBUSD(corr[kk], x128, _mm256_load_si256((const __m256i *)(wp + 32 * kk)));
        }

        for (int b = 0; b < B; b += 2) {
            const int two = b + 1 < B;
            const uint8_t *xa = k->xu + (size_t)b * cols20, *xb = two ? xa + cols20 : xa;
            __m256i a[4], c[4];
            for (int kk = 0; kk < 4; kk++) a[kk] = c[kk] = _mm256_setzero_si256();
            for (int j = 0; j < nblk; j++) {
                const int8_t *wp = W4 + (size_t)j * 128;
                int32_t ia, ib; memcpy(&ia, xa + 4 * j, 4); memcpy(&ib, xb + 4 * j, 4);
                __m256i va = _mm256_set1_epi32(ia), vb = _mm256_set1_epi32(ib);
                __m256i w0 = _mm256_load_si256((const __m256i *)(wp)),      w1 = _mm256_load_si256((const __m256i *)(wp + 32));
                __m256i w2 = _mm256_load_si256((const __m256i *)(wp + 64)), w3 = _mm256_load_si256((const __m256i *)(wp + 96));
                a[0] = DPBUSD(a[0], va, w0); a[1] = DPBUSD(a[1], va, w1); a[2] = DPBUSD(a[2], va, w2); a[3] = DPBUSD(a[3], va, w3);
                c[0] = DPBUSD(c[0], vb, w0); c[1] = DPBUSD(c[1], vb, w1); c[2] = DPBUSD(c[2], vb, w2); c[3] = DPBUSD(c[3], vb, w3);
            }
            gemm_store(k, tile, b, a, corr);
            if (two) gemm_store(k, tile, b + 1, c, corr);
        }
    }
    _mm_free(W4);
    free(rowbuf);
}

void aim_t3_gemm_simd(const aim_t3_tiled *t, const float *X, int ldx, int B, float *Y, int ldy)
{
    const int cols = t->cols, cols20 = (t->cols_pad + 19) / 20 * 20;
    uint8_t *xu = malloc((size_t)B * cols20);
    float *sx = malloc((size_t)B * sizeof(float));
    for (int b = 0; b < B; b++) {
        const float *x = X + (size_t)b * ldx;
        uint8_t *u = xu + (size_t)b * cols20;
        float amax = 0;
        for (int c = 0; c < cols; c++) { float a = fabsf(x[c]); if (a > amax) amax = a; }
        float s = amax > 0 ? amax / 127.0f : 1.0f, inv = 1.0f / s;
        sx[b] = s;
        for (int c = 0; c < cols; c++) { float v = x[c] * inv; int q = (int)(v >= 0 ? v + 0.5f : v - 0.5f); u[c] = (uint8_t)(q + 128); }
        for (int c = cols; c < cols20; c++) u[c] = 128;
    }
    gemm_task k = { t, xu, sx, B, cols20, Y, ldy, 0 };
    aim_pool_run(gemm_task_fn, &k);
    free(xu); free(sx);
}

#else
void aim_t3_gemm_simd(const aim_t3_tiled *t, const float *X, int ldx, int B, float *Y, int ldy)
{
    for (int b = 0; b < B; b++) aim_t3_gemv_simd(t, X + (size_t)b * ldx, Y + (size_t)b * ldy);
}
#endif
