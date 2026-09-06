/* Codici a byte generici: n pesi interi piccoli in un byte, con vincoli.
 *
 * Il caso nuovo e' il codice enumerativo sparso "sNK": n pesi ternari con
 * al piu' K non nulli. Il numero di configurazioni e'  sum_{j<=K} C(n,j) 2^j:
 *   s36 -> 233 <= 256  : 6 pesi in un byte, 1.33 bit/peso
 *   s52 -> 51          : (5 pesi, <=2 non nulli) 6 bit
 * E' sparsita' strutturata a lunghezza fissa: si pota il modello a <=K su n,
 * poi ogni gruppo costa esattamente un byte. Nessun indice, nessuna maschera. */
#include "aim.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

static int ipow(int b, int e) { int r = 1; while (e-- > 0) r *= b; return r; }

/* indice base-`levels` della n-upla (valore + lev/2 per cifra) */
static int tuple_index(const aim_code *c, const int8_t *v)
{
    int idx = 0, off = c->levels / 2;
    for (int i = c->n - 1; i >= 0; i--) idx = idx * c->levels + (v[i] + off);
    return idx;
}

int aim_code_encode(const aim_code *c, const int8_t *vals) { return c->enc[tuple_index(c, vals)]; }

int aim_code_init(aim_code *c, const char *name)
{
    memset(c, 0, sizeof *c);
    c->name = name;
    if (!strcmp(name, "t5"))      { c->n = 5; c->levels = 3; c->kmax = 5; }
    else if (!strcmp(name, "q4")) { c->n = 4; c->levels = 4; c->kmax = 4; }
    else if (name[0] == 's' && name[1] >= '1' && name[1] <= '8' && name[2] >= '0' && name[2] <= '8' && !name[3]) {
        c->n = name[1] - '0'; c->levels = 3; c->kmax = name[2] - '0';
        if (c->kmax > c->n) return -1;
    } else return -1;

    int total = ipow(c->levels, c->n);
    c->enc = malloc(total * sizeof(int16_t));
    int off = c->levels / 2, ncodes = 0;
    for (int idx = 0; idx < total; idx++) {
        int8_t v[AIM_CODE_MAXW]; int t = idx, nz = 0;
        for (int i = 0; i < c->n; i++) { v[i] = (int8_t)(t % c->levels - off); t /= c->levels; nz += v[i] != 0; }
        if (nz > c->kmax || ncodes >= 256) { c->enc[idx] = -1; continue; }
        /* per t5 il codice deve coincidere con la base 3 nativa (idx stesso) */
        c->enc[idx] = (int16_t)ncodes;
        memcpy(c->dec[ncodes], v, c->n);
        ncodes++;
    }
    c->ncodes = ncodes;
    c->bits_per_weight = 8.0 / c->n;
    if (ncodes < total && c->kmax == c->n) { aim_code_free(c); return -1; }   /* non sta in un byte */
    return 0;
}

void aim_code_free(aim_code *c) { free(c->enc); c->enc = NULL; }

static void tiled_alloc(const aim_code *c, int rows, int cols, aim_t3_tiled *t)
{
    memset(t, 0, sizeof *t);
    t->code = c; t->rows = rows; t->cols = cols;
    t->cols_pad = (cols + c->n - 1) / c->n * c->n;
    t->G = t->cols_pad / c->n;
    t->rows_pad = (rows + AIM_T3_TILE - 1) / AIM_T3_TILE * AIM_T3_TILE;
    t->data = calloc((size_t)t->rows_pad * t->G, 1);
    t->scale = calloc(t->rows_pad, sizeof(float));
}

static inline uint8_t *tile_byte(aim_t3_tiled *t, int r, int g)
{
    return t->data + (((size_t)(r / AIM_T3_TILE) * t->G + g) * AIM_T3_TILE + r % AIM_T3_TILE);
}

/* Pota una n-upla a <= kmax non nulli togliendo i |mag| piu' piccoli. Ritorna quanti potati. */
static int prune_group(const aim_code *c, int8_t *v, const float *mag)
{
    int nz = 0; for (int i = 0; i < c->n; i++) nz += v[i] != 0;
    int pruned = 0;
    while (nz > c->kmax) {
        int best = 0; float bm = 1e30f;
        for (int i = 0; i < c->n; i++) if (v[i] != 0 && mag[i] < bm) { bm = mag[i]; best = i; }
        v[best] = 0; nz--; pruned++;
    }
    return pruned;
}

int aim_code_pack_ternary(const aim_code *c, const int8_t *T, const float *mag, int rows, int cols,
                          const float *row_scale, aim_t3_tiled *t, long *pruned)
{
    tiled_alloc(c, rows, cols, t);
    long np = 0;
    int8_t v[AIM_CODE_MAXW]; float mg[AIM_CODE_MAXW];
    for (int r = 0; r < rows; r++) {
        t->scale[r] = row_scale[r];
        for (int g = 0; g < t->G; g++) {
            for (int i = 0; i < c->n; i++) {
                int col = g * c->n + i;
                v[i] = col < cols ? T[(size_t)r * cols + col] : 0;
                mg[i] = col < cols && mag ? mag[(size_t)r * cols + col] : 0;
            }
            np += prune_group(c, v, mg);
            int code = aim_code_encode(c, v);
            if (code < 0) return -1;
            *tile_byte(t, r, g) = (uint8_t)code;
        }
    }
    if (pruned) *pruned = np;
    return 0;
}

/* Quantizzazione da float: TWN per riga (delta = 0.7 mean|w|), livelli = round(w/scale) clampati */
int aim_code_quantize(const aim_code *c, const float *W, int rows, int cols, aim_t3_tiled *t)
{
    int8_t *T = malloc((size_t)rows * cols);
    float *sc = malloc(rows * sizeof(float));
    const int lmax = c->levels / 2;
    for (int r = 0; r < rows; r++) {
        const float *w = W + (size_t)r * cols;
        double s = 0; for (int i = 0; i < cols; i++) s += fabsf(w[i]);
        float mean = (float)(s / cols);
        float scale;
        if (c->levels == 3) {
            float delta = 0.7f * mean; double s_nz = 0; int n_nz = 0;
            for (int i = 0; i < cols; i++) if (fabsf(w[i]) > delta) { s_nz += fabsf(w[i]); n_nz++; }
            scale = n_nz ? (float)(s_nz / n_nz) : 1.0f;
            for (int i = 0; i < cols; i++) T[(size_t)r * cols + i] = w[i] > delta ? 1 : (w[i] < -delta ? -1 : 0);
        } else {
            scale = mean * 2.0f / lmax + 1e-8f;    /* passo uniforme: i livelli coprono ~2 mean|w| */
            for (int i = 0; i < cols; i++) {
                float q = w[i] / scale; int v = (int)(q >= 0 ? q + 0.5f : q - 0.5f);
                int hi = c->levels % 2 == 0 ? lmax - 1 : lmax;   /* pari: [-lmax, lmax-1] */
                if (v > hi) v = hi;
                if (v < -lmax) v = -lmax;
                T[(size_t)r * cols + i] = (int8_t)v;
            }
        }
        sc[r] = scale;
    }
    /* magnitudini = |w| per la potatura */
    float *mag = malloc((size_t)rows * cols * sizeof(float));
    for (size_t i = 0; i < (size_t)rows * cols; i++) mag[i] = fabsf(W[i]);
    int rc = aim_code_pack_ternary(c, T, mag, rows, cols, sc, t, NULL);
    free(T); free(sc); free(mag);
    return rc;
}

void aim_code_unpack_rows(const aim_t3_tiled *t, int r0, int nrows, int8_t *out, int ld)
{
    const aim_code *c = t->code;
    for (int r = r0; r < r0 + nrows; r++) {
        int8_t *o = out + (size_t)(r - r0) * ld;
        const uint8_t *row = t->data + ((size_t)(r / AIM_T3_TILE) * t->G) * AIM_T3_TILE + r % AIM_T3_TILE;
        for (int g = 0; g < t->G; g++) {
            const int8_t *d = c->dec[row[(size_t)g * AIM_T3_TILE]];
            for (int i = 0; i < c->n; i++) o[g * c->n + i] = d[i];
        }
    }
}

/* LUT scalare generica con blocking: tabella T[G][ncodes] costruita dalle attivazioni */
typedef struct { const aim_t3_tiled *t; const float *x; float *y; float *T; atomic_int next_g, next_r; } lut_task;

static void lut_task_fn(void *ctx, int tid, int nth)
{
    (void)tid; (void)nth;
    lut_task *k = ctx;
    const aim_t3_tiled *t = k->t; const aim_code *c = t->code;
    const int G = t->G, NC = c->ncodes;
    for (int g0; (g0 = aim_pool_next(&k->next_g, 16)) < G;)
        for (int g = g0; g < g0 + 16 && g < G; g++) {
            float xg[AIM_CODE_MAXW];
            for (int i = 0; i < c->n; i++) { int col = g * c->n + i; xg[i] = col < t->cols ? k->x[col] : 0; }
            float *Tg = k->T + (size_t)g * NC;
            for (int b = 0; b < NC; b++) { float s = 0; for (int i = 0; i < c->n; i++) s += c->dec[b][i] * xg[i]; Tg[b] = s; }
        }
    aim_pool_barrier();
    const int ntiles = t->rows_pad / AIM_T3_TILE;
    for (int tile; (tile = aim_pool_next(&k->next_r, 1)) < ntiles;) {
        const uint8_t *td = t->data + (size_t)tile * G * AIM_T3_TILE;
        float acc[AIM_T3_TILE]; memset(acc, 0, sizeof acc);
        for (int g = 0; g < G; g++) {
            const float *Tg = k->T + (size_t)g * NC; const uint8_t *cd = td + (size_t)g * AIM_T3_TILE;
            for (int i = 0; i < AIM_T3_TILE; i++) acc[i] += Tg[cd[i]];
        }
        for (int i = 0; i < AIM_T3_TILE; i++) { int r = tile * AIM_T3_TILE + i; if (r < t->rows) k->y[r] = acc[i] * t->scale[r]; }
    }
}

void aim_code_gemv_lut(const aim_t3_tiled *t, const float *x, float *y)
{
    lut_task k = { t, x, y, malloc((size_t)t->G * t->code->ncodes * sizeof(float)), 0, 0 };
    aim_pool_run(lut_task_fn, &k);
    free(k.T);
}
