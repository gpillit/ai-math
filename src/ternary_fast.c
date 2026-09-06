/* Kernel veloci per la matrice ternaria in base 3.
 *
 * A) blocked : T[243] per gruppo, ma si processano 32 gruppi (31 KB, in L1)
 *              per volta su un blocco di righe.
 * B) factored: Z_3^5 = Z_3^3 x Z_3^2  ->  T[b] = Tlo[b mod 27] + Thi[b / 27].
 *              36 voci per gruppo invece di 243: la tabella completa di una
 *              riga da 4096 colonne pesa 118 KB invece di 796 KB.
 * C) simd    : come B, ma attivazioni int8, tabelle int16 divise in byte
 *              basso/alto e lookup con vpshufb su 32 righe alla volta (tile).
 */
#include "aim.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* T[g][243] costruita in modo incrementale dalla base 3 */
static void build_T243(const aim_t3_mat *m, const float *x, float *T)
{
    const int G = m->bytes_per_row;
    #pragma omp parallel for
    for (int g = 0; g < G; g++) {
        float xg[AIM_T3_GROUP];
        for (int i = 0; i < AIM_T3_GROUP; i++) {
            int c = g * AIM_T3_GROUP + i;
            xg[i] = c < m->cols ? x[c] : 0.0f;
        }
        float *Tg = T + (size_t)g * AIM_T3_CODES;
        Tg[0] = -(xg[0] + xg[1] + xg[2] + xg[3] + xg[4]);
        int span = 1;
        for (int i = 0; i < AIM_T3_GROUP; i++) {
            for (int b = 0; b < span; b++) {
                Tg[b + span]     = Tg[b] + xg[i];
                Tg[b + 2 * span] = Tg[b] + 2 * xg[i];
            }
            span *= 3;
        }
    }
}

/* ------------------------------ A ---------------------------------- */
void aim_t3_gemv_lut_blocked(const aim_t3_mat *m, const float *x, float *y)
{
    const int G = m->bytes_per_row, R = m->rows;
    float *T = malloc((size_t)G * AIM_T3_CODES * sizeof(float));
    build_T243(m, x, T);

    const int GB = 32;   /* 32 gruppi * 972 B = 31 KB: sta in L1 */
    const int RB = 64;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int r0 = 0; r0 < R; r0 += RB) {
        int r1 = r0 + RB < R ? r0 + RB : R;
        float acc[64];
        memset(acc, 0, sizeof acc);
        for (int g0 = 0; g0 < G; g0 += GB) {
            int g1 = g0 + GB < G ? g0 + GB : G;
            const float *Tb = T + (size_t)g0 * AIM_T3_CODES;
            for (int r = r0; r < r1; r++) {
                const uint8_t *row = m->data + (size_t)r * G + g0;
                float a0 = 0, a1 = 0;
                int n = g1 - g0, g = 0;
                for (; g + 2 <= n; g += 2) {
                    a0 += Tb[(size_t)(g + 0) * AIM_T3_CODES + row[g + 0]];
                    a1 += Tb[(size_t)(g + 1) * AIM_T3_CODES + row[g + 1]];
                }
                for (; g < n; g++) a0 += Tb[(size_t)g * AIM_T3_CODES + row[g]];
                acc[r - r0] += a0 + a1;
            }
        }
        for (int r = r0; r < r1; r++) y[r] = m->scale[r] * acc[r - r0];
    }
    free(T);
}

/* ------------------------------ B ---------------------------------- */
/* Tabelle fattorizzate: per gruppo 27 + 9 float, padding a 32 + 16 = 48 float */
#define FAC_STRIDE 48
static void build_Tfac(const aim_t3_mat *m, const float *x, float *T)
{
    const int G = m->bytes_per_row;
    #pragma omp parallel for
    for (int g = 0; g < G; g++) {
        float xg[AIM_T3_GROUP];
        for (int i = 0; i < AIM_T3_GROUP; i++) {
            int c = g * AIM_T3_GROUP + i;
            xg[i] = c < m->cols ? x[c] : 0.0f;
        }
        float *lo = T + (size_t)g * FAC_STRIDE, *hi = lo + 32;
        /* lo: trit 0,1,2 ; hi: trit 3,4 : stessa costruzione incrementale */
        lo[0] = -(xg[0] + xg[1] + xg[2]);
        int span = 1;
        for (int i = 0; i < 3; i++) {
            for (int b = 0; b < span; b++) { lo[b + span] = lo[b] + xg[i]; lo[b + 2 * span] = lo[b] + 2 * xg[i]; }
            span *= 3;
        }
        for (int b = 27; b < 32; b++) lo[b] = 0;
        hi[0] = -(xg[3] + xg[4]);
        span = 1;
        for (int i = 3; i < 5; i++) {
            for (int b = 0; b < span; b++) { hi[b + span] = hi[b] + xg[i]; hi[b + 2 * span] = hi[b] + 2 * xg[i]; }
            span *= 3;
        }
        for (int b = 9; b < 16; b++) hi[b] = 0;
    }
}

void aim_t3_gemv_lut_factored(const aim_t3_mat *m, const float *x, float *y)
{
    const int G = m->bytes_per_row, R = m->rows;
    float *T = malloc((size_t)G * FAC_STRIDE * sizeof(float));
    build_Tfac(m, x, T);

    #pragma omp parallel for schedule(dynamic, 64)
    for (int r = 0; r < R; r++) {
        const uint8_t *row = m->data + (size_t)r * G;
        float a0 = 0, a1 = 0;
        for (int g = 0; g < G; g++) {
            unsigned b = row[g];
            unsigned hi = (b * 19u) >> 9;          /* b / 27 esatto per b < 243 */
            unsigned lo = b - 27u * hi;
            const float *Tg = T + (size_t)g * FAC_STRIDE;
            a0 += Tg[lo];
            a1 += Tg[32 + hi];
        }
        y[r] = m->scale[r] * (a0 + a1);
    }
    free(T);
}

/* ------------------------------ C ---------------------------------- */
int aim_t3_tile(const aim_t3_mat *m, aim_t3_tiled *t)
{
    memset(t, 0, sizeof *t);      /* code = NULL: ternario base 3 nativo */
    t->rows = m->rows; t->cols = m->cols; t->cols_pad = m->cols_pad; t->G = m->bytes_per_row;
    t->rows_pad = (m->rows + AIM_T3_TILE - 1) / AIM_T3_TILE * AIM_T3_TILE;
    size_t n = (size_t)t->rows_pad * t->G;
    t->data  = calloc(n ? n : 1, 1);
    t->scale = calloc(t->rows_pad, sizeof(float));
    if (!t->data || !t->scale) return -1;
    for (int r = 0; r < m->rows; r++) {
        int tile = r / AIM_T3_TILE, i = r % AIM_T3_TILE;
        const uint8_t *src = m->data + (size_t)r * t->G;
        uint8_t *dst = t->data + (size_t)tile * t->G * AIM_T3_TILE + i;
        for (int g = 0; g < t->G; g++) dst[(size_t)g * AIM_T3_TILE] = src[g];
        t->scale[r] = m->scale[r];
    }
    return 0;
}

void aim_t3_tiled_free(aim_t3_tiled *t)
{
    free(t->data); free(t->scale); t->data = NULL; t->scale = NULL;
}

#if defined(__AVX2__)
#include <immintrin.h>

/* Per gruppo, 96 byte di tabelle per vpshufb:
 *  [0]  Tlo[0..15]  byte basso   [16] Tlo[0..15]  byte alto
 *  [32] Tlo[16..31] byte basso   [48] Tlo[16..31] byte alto
 *  [64] Thi[0..15]  byte basso   [80] Thi[0..15]  byte alto      */
#define SIMD_TBL 96

/* tabelle di un gruppo da xq (int8): Tlo[27] e Thi[9] int16, spezzate in byte */
static inline void build_group_tables(const int8_t *xg, uint8_t *d)
{
    int16_t lo[32] = {0}, hi[16] = {0};
    lo[0] = -(xg[0] + xg[1] + xg[2]);
    int span = 1;
    for (int i = 0; i < 3; i++) {
        for (int b = 0; b < span; b++) { lo[b + span] = lo[b] + xg[i]; lo[b + 2 * span] = lo[b] + 2 * xg[i]; }
        span *= 3;
    }
    hi[0] = -(xg[3] + xg[4]);
    span = 1;
    for (int i = 3; i < 5; i++) {
        for (int b = 0; b < span; b++) { hi[b + span] = hi[b] + xg[i]; hi[b + 2 * span] = hi[b] + 2 * xg[i]; }
        span *= 3;
    }
    for (int i = 0; i < 16; i++) {
        d[i]      = (uint8_t)(lo[i] & 0xFF);       d[16 + i] = (uint8_t)((lo[i] >> 8) & 0xFF);
        d[32 + i] = (uint8_t)(lo[16 + i] & 0xFF);  d[48 + i] = (uint8_t)((lo[16 + i] >> 8) & 0xFF);
        d[64 + i] = (uint8_t)(hi[i] & 0xFF);       d[80 + i] = (uint8_t)((hi[i] >> 8) & 0xFF);
    }
}

/* Lookup nel dominio byte: 32 indici per vpshufb.
 * Tabella da 32 voci = due vpshufb (A: 0..15, B: 16..31) selezionati
 * azzerando il risultato non valido tramite il bit 7 dell'indice. */
static inline void lut32_bytes(__m256i idx, __m256i A_lo, __m256i A_hi, __m256i B_lo, __m256i B_hi,
                               __m256i c15, __m256i c16, __m256i c80, __m256i *out_lo, __m256i *out_hi)
{
    __m256i m  = _mm256_cmpgt_epi8(idx, c15);              /* 0xFF dove idx >= 16 */
    __m256i ia = _mm256_or_si256(idx, m);                  /* bit7 set se >= 16 -> 0 */
    __m256i ib = _mm256_or_si256(_mm256_xor_si256(idx, c16), _mm256_andnot_si256(m, c80));
    *out_lo = _mm256_or_si256(_mm256_shuffle_epi8(A_lo, ia), _mm256_shuffle_epi8(B_lo, ib));
    *out_hi = _mm256_or_si256(_mm256_shuffle_epi8(A_hi, ia), _mm256_shuffle_epi8(B_hi, ib));
}

/* Un gruppo (32 byte = 32 righe x 5 pesi) di un tile: accumula in acc16_0/1 */
#define T3_GROUP_STEP(codes, acc0, acc1)                                                          \
    do {                                                                                           \
        __m256i w0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(codes));                         \
        __m256i w1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(codes, 1));                    \
        __m256i h0 = _mm256_mulhi_epu16(w0, c2432), h1 = _mm256_mulhi_epu16(w1, c2432);           \
        __m256i l0 = _mm256_sub_epi16(w0, _mm256_mullo_epi16(h0, c27));                           \
        __m256i l1 = _mm256_sub_epi16(w1, _mm256_mullo_epi16(h1, c27));                           \
        __m256i lo = _mm256_packus_epi16(l0, l1), hi = _mm256_packus_epi16(h0, h1);               \
        __m256i vlo8, vhi8;                                                                        \
        lut32_bytes(lo, A_lo, A_hi, B_lo, B_hi, c15, c16, c80, &vlo8, &vhi8);                     \
        __m256i ulo8 = _mm256_shuffle_epi8(H_lo, hi), uhi8 = _mm256_shuffle_epi8(H_hi, hi);        \
        acc0 = _mm256_add_epi16(acc0, _mm256_add_epi16(_mm256_unpacklo_epi8(vlo8, vhi8),           \
                                                       _mm256_unpacklo_epi8(ulo8, uhi8)));         \
        acc1 = _mm256_add_epi16(acc1, _mm256_add_epi16(_mm256_unpackhi_epi8(vlo8, vhi8),           \
                                                       _mm256_unpackhi_epi8(ulo8, uhi8)));         \
    } while (0)

#define T3_FOLD(acc16, a32a, a32b)                                                                 \
    do {                                                                                           \
        a32a = _mm256_add_epi32(a32a, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc16)));      \
        a32b = _mm256_add_epi32(a32b, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc16, 1))); \
        acc16 = _mm256_setzero_si256();                                                            \
    } while (0)

static inline void t3_store_tile(const aim_t3_tiled *t, float sx, int r0,
                                 __m256i a0, __m256i a1, __m256i a2, __m256i a3, float *y)
{
    __m256i accs[4] = { a0, a1, a2, a3 };
    for (int i = 0; i < 4; i++) {
        int r = r0 + 8 * i;
        if (r >= t->rows) break;
        __m256 f = _mm256_cvtepi32_ps(accs[i]);
        __m256 s = _mm256_mul_ps(_mm256_loadu_ps(t->scale + r), _mm256_set1_ps(sx));
        f = _mm256_mul_ps(f, s);
        if (r + 8 <= t->rows) _mm256_storeu_ps(y + r, f);
        else { float tmp[8]; _mm256_storeu_ps(tmp, f); for (int k = 0; r + k < t->rows; k++) y[r + k] = tmp[k]; }
    }
}

static int t3_chunk = 0, T3_PF = 24;   /* iterazioni per chunk; gruppi di prefetch (24*32 B = 768 B) */

typedef struct {
    const aim_t3_tiled *t;
    const int8_t *xq;
    uint8_t *TB;
    float *y;
    float sx;
    atomic_int next_group, next_tile;
} t3_task;

static void t3_task_fn(void *ctx, int tid, int nth)
{
    (void)tid;
    t3_task *k = ctx;
    const aim_t3_tiled *t = k->t;
    const int G = t->G;
    uint8_t *TB = k->TB;

    /* fase 1: tabelle per gruppo (dinamico, chunk di 32 gruppi) */
    for (int g0; (g0 = aim_pool_next(&k->next_group, 32)) < G;) {
        int g1 = g0 + 32 < G ? g0 + 32 : G;
        for (int g = g0; g < g1; g++)
            build_group_tables(k->xq + g * AIM_T3_GROUP, TB + (size_t)g * SIMD_TBL);
    }
    aim_pool_barrier();

    const __m256i c2432 = _mm256_set1_epi16(2432);   /* mulhi(b, 19<<7) = b/27 per b < 243 */
    const __m256i c27   = _mm256_set1_epi16(27);
    const __m256i c15   = _mm256_set1_epi8(15), c16 = _mm256_set1_epi8(16), c80 = _mm256_set1_epi8((char)0x80);
    const int KFOLD = 48;  /* 48 * (381+254) < 32767: accumulo int16 sicuro */
    const int ntiles = t->rows_pad / AIM_T3_TILE;
    const size_t tile_bytes = (size_t)G * AIM_T3_TILE;
    const float sx = k->sx;
    float *y = k->y;

    /* fase 2: coppie di tile (dinamico). Due tile per iterazione: le 6
     * tabelle caricate servono 64 righe; chunk grosso = tratto contiguo di RAM */
    /* chunk adattivo: ~6 chunk per thread, cosi' la coda (E-core lenti) e' corta */
    int step = t3_chunk > 0 ? 2 * t3_chunk : ntiles / (nth * 6);
    step = step < 2 ? 2 : step > 16 ? 16 : (step & ~1);
    for (int tp0; (tp0 = aim_pool_next(&k->next_tile, step)) < ntiles;) {
        int tp1 = tp0 + step < ntiles ? tp0 + step : ntiles;
        for (int tp = tp0; tp < tp1; tp += 2) {
            const int two = tp + 1 < ntiles;
            const uint8_t *td0 = t->data + (size_t)tp * tile_bytes;
            const uint8_t *td1 = two ? td0 + tile_bytes : td0;
            __m256i p0 = _mm256_setzero_si256(), p1 = p0, p2 = p0, p3 = p0;   /* tile 0, int32 */
            __m256i q0 = p0, q1 = p0, q2 = p0, q3 = p0;                       /* tile 1, int32 */
            __m256i pa = p0, pb = p0, qa = p0, qb = p0;                       /* int16 */
            int since = 0;
            for (int g = 0; g < G; g++) {
                const uint8_t *tb = TB + (size_t)g * SIMD_TBL;
                __m256i A_lo = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)(tb + 0)));
                __m256i A_hi = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)(tb + 16)));
                __m256i B_lo = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)(tb + 32)));
                __m256i B_hi = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)(tb + 48)));
                __m256i H_lo = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)(tb + 64)));
                __m256i H_hi = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)(tb + 80)));
                _mm_prefetch((const char *)(td0 + (size_t)(g + T3_PF) * AIM_T3_TILE), _MM_HINT_T0);
                __m256i c0 = _mm256_loadu_si256((const __m256i *)(td0 + (size_t)g * AIM_T3_TILE));
                T3_GROUP_STEP(c0, pa, pb);
                if (two) {
                    _mm_prefetch((const char *)(td1 + (size_t)(g + T3_PF) * AIM_T3_TILE), _MM_HINT_T0);
                    __m256i c1 = _mm256_loadu_si256((const __m256i *)(td1 + (size_t)g * AIM_T3_TILE));
                    T3_GROUP_STEP(c1, qa, qb);
                }
                if (++since == KFOLD || g == G - 1) {
                    T3_FOLD(pa, p0, p1); T3_FOLD(pb, p2, p3);
                    T3_FOLD(qa, q0, q1); T3_FOLD(qb, q2, q3);
                    since = 0;
                }
            }
            t3_store_tile(t, sx, tp * AIM_T3_TILE, p0, p1, p2, p3, y);
            if (two) t3_store_tile(t, sx, (tp + 1) * AIM_T3_TILE, q0, q1, q2, q3, y);
        }
    }
}

void aim_t3_gemv_simd(const aim_t3_tiled *t, const float *x, float *y)
{
    if (t->code) { aim_code_gemv_lut(t, x, y); return; }   /* codice generico: LUT scalare */
    const int G = t->G, cols = t->cols;
    if (!t3_chunk) {
        const char *e = getenv("AIM_CHUNK"); t3_chunk = e ? atoi(e) : -1;   /* -1: adattivo */
        e = getenv("AIM_PF"); if (e) T3_PF = atoi(e);
    }

    /* attivazioni int8 simmetriche per-tensore */
    float amax = 0;
    for (int c = 0; c < cols; c++) { float a = fabsf(x[c]); if (a > amax) amax = a; }
    const float sx = amax > 0 ? amax / 127.0f : 1.0f, inv = 1.0f / sx;
    int8_t *xq = calloc(t->cols_pad, 1);
    for (int c = 0; c < cols; c++) { float v = x[c] * inv; xq[c] = (int8_t)(v >= 0 ? v + 0.5f : v - 0.5f); }

    t3_task k = { t, xq, _mm_malloc((size_t)G * SIMD_TBL, 32), y, sx, 0, 0 };
    aim_pool_run(t3_task_fn, &k);
    _mm_free(k.TB);
    free(xq);
}
#else
void aim_t3_gemv_simd(const aim_t3_tiled *t, const float *x, float *y)
{
    /* fallback: ricostruisce la matrice row-major e usa il kernel fattorizzato */
    aim_t3_mat m = { t->rows, t->cols, t->cols_pad, t->G, NULL, t->scale };
    m.data = malloc((size_t)t->rows * t->G);
    for (int r = 0; r < t->rows; r++)
        for (int g = 0; g < t->G; g++)
            m.data[(size_t)r * t->G + g] = t->data[((size_t)(r / AIM_T3_TILE) * t->G + g) * AIM_T3_TILE + r % AIM_T3_TILE];
    aim_t3_gemv_lut_factored(&m, x, y);
    free(m.data);
}
#endif
