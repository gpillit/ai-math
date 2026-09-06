#include "aim_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ATTN_CHUNK 256   /* posizioni per item nell'attention */

/* ---------------------------------------------------------------- load */
static int read_t3(const uint8_t **p, const uint8_t *end, aim_t3_tiled *t)
{
    if (*p + 8 > end) return -1;
    int32_t rows, cols;
    memcpy(&rows, *p, 4); memcpy(&cols, *p + 4, 4); *p += 8;
    t->rows = rows; t->cols = cols;
    t->cols_pad = (cols + AIM_T3_GROUP - 1) / AIM_T3_GROUP * AIM_T3_GROUP;
    t->G = t->cols_pad / AIM_T3_GROUP;
    t->rows_pad = (rows + AIM_T3_TILE - 1) / AIM_T3_TILE * AIM_T3_TILE;
    size_t nd = (size_t)t->rows_pad * t->G, ns = (size_t)t->rows_pad * sizeof(float);
    if (*p + nd + ns > end) return -1;
    t->data = (uint8_t *)*p; *p += nd;
    t->scale = (float *)*p; *p += ns;
    return 0;
}

static const float *read_f32(const uint8_t **p, size_t n) { const float *r = (const float *)*p; *p += n * 4; return r; }

int aim_model_load(const char *path, aim_model *m)
{
    memset(m, 0, sizeof *m);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "impossibile aprire %s\n", path); return -1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    m->blob = malloc(sz); m->blob_size = sz;
    if (!m->blob || fread(m->blob, 1, sz, f) != (size_t)sz) { fclose(f); return -1; }
    fclose(f);

    const uint8_t *p = m->blob, *end = m->blob + sz;
    if (memcmp(p, "AIMODEL1", 8) != 0) { fprintf(stderr, "magic errato\n"); return -1; }
    p += 8;
    int32_t hdr[8]; memcpy(hdr, p, 32); p += 32;
    m->n_layers = hdr[0]; m->hidden = hdr[1]; m->inter = hdr[2]; m->n_heads = hdr[3];
    m->n_kv = hdr[4]; m->head_dim = hdr[5]; m->vocab = hdr[6]; m->max_pos = hdr[7];
    memcpy(&m->eps, p, 4); memcpy(&m->rope_theta, p + 4, 4); p += 8;

    m->emb_q = (const int8_t *)p; p += (size_t)m->vocab * m->hidden;
    m->emb_s = read_f32(&p, m->vocab);
    m->final_norm = read_f32(&p, m->hidden);
    m->layers = calloc(m->n_layers, sizeof(aim_layer));
    for (int l = 0; l < m->n_layers; l++) {
        aim_layer *L = &m->layers[l];
        L->ln1 = read_f32(&p, m->hidden);
        if (read_t3(&p, end, &L->qkv)) return -1;
        L->attn_sub = read_f32(&p, m->hidden);
        if (read_t3(&p, end, &L->o)) return -1;
        L->ln2 = read_f32(&p, m->hidden);
        if (read_t3(&p, end, &L->gate_up)) return -1;
        L->ffn_sub = read_f32(&p, m->inter);
        if (read_t3(&p, end, &L->down)) return -1;
        m->ternary_bytes += (size_t)L->qkv.rows_pad * L->qkv.G + (size_t)L->o.rows_pad * L->o.G
                          + (size_t)L->gate_up.rows_pad * L->gate_up.G + (size_t)L->down.rows_pad * L->down.G;
    }
    if (p != end) { fprintf(stderr, "file: %ld byte inattesi\n", (long)(end - p)); return -1; }
    if ((m->n_heads / m->n_kv) % 4 != 0 || m->head_dim % 8 != 0) { fprintf(stderr, "attention: servono group%%4==0 e head_dim%%8==0\n"); return -1; }
    return 0;
}

void aim_model_free(aim_model *m) { free(m->layers); free(m->blob); memset(m, 0, sizeof *m); }

/* ---------------------------------------------------------------- ctx */
int aim_ctx_init(aim_ctx *c, const aim_model *m, int max_ctx)
{
    memset(c, 0, sizeof *c);
    c->m = m; c->max_ctx = max_ctx; c->pos = 0;
    int H = m->hidden, kvd = m->n_kv * m->head_dim, half = m->head_dim / 2;
    c->k_cache = calloc((size_t)m->n_layers * max_ctx * kvd, sizeof(float));
    c->v_cache = calloc((size_t)m->n_layers * max_ctx * kvd, sizeof(float));
    c->x = malloc(H * sizeof(float));  c->xb = malloc(H * sizeof(float));
    c->qkv = malloc((H + 2 * kvd) * sizeof(float));
    c->att = malloc((size_t)m->n_heads * max_ctx * sizeof(float));
    c->part = malloc((size_t)m->n_heads * ((max_ctx + ATTN_CHUNK - 1) / ATTN_CHUNK) * m->head_dim * sizeof(float));
    c->attn = malloc(H * sizeof(float));
    c->gu = malloc(2 * (size_t)m->inter * sizeof(float));
    c->h = malloc((size_t)m->inter * sizeof(float));
    c->logits = malloc((size_t)m->vocab * sizeof(float));
    c->rope_cos = malloc((size_t)max_ctx * half * sizeof(float));
    c->rope_sin = malloc((size_t)max_ctx * half * sizeof(float));
    if (!c->k_cache || !c->v_cache || !c->logits || !c->rope_cos) return -1;
    for (int p = 0; p < max_ctx; p++)
        for (int i = 0; i < half; i++) {
            double inv = pow((double)m->rope_theta, -2.0 * i / m->head_dim);
            c->rope_cos[(size_t)p * half + i] = (float)cos(p * inv);
            c->rope_sin[(size_t)p * half + i] = (float)sin(p * inv);
        }
    return 0;
}

void aim_ctx_free(aim_ctx *c)
{
    free(c->k_cache); free(c->v_cache); free(c->x); free(c->xb); free(c->qkv); free(c->att); free(c->part);
    free(c->attn); free(c->gu); free(c->h); free(c->logits); free(c->rope_cos); free(c->rope_sin);
    memset(c, 0, sizeof *c);
}

/* ---------------------------------------------------------------- ops */
static void rmsnorm(float *out, const float *x, const float *w, int n, float eps)
{
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float r = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

/* rotate_half (stile Llama/HF): coppie (i, i+half) */
static void rope(float *v, int n_heads, int hd, const float *cs, const float *sn)
{
    int half = hd / 2;
    for (int h = 0; h < n_heads; h++) {
        float *p = v + h * hd;
        for (int i = 0; i < half; i++) {
            float a = p[i], b = p[i + half];
            p[i]        = a * cs[i] - b * sn[i];
            p[i + half] = b * cs[i] + a * sn[i];
        }
    }
}

typedef struct {
    aim_ctx *c;
    const float *q;   int ldq;     /* query del token b: q + b*ldq + h*HD */
    float *out;       int ldo;     /* output: out + b*ldo + h*HD */
    float *sc;                     /* scratch punteggi/pesi: (b*NH + h) * max_ctx */
    float *part;                   /* scratch parziali: ((b*NH + h) * nchunk + ch) * HD */
    const float *kbase, *vbase;
    int pos0, B, kvd, HD, NH, group, nchunk, chunk; float scale;
    atomic_int next1, next2, next3;
} attn_task;

/* Semianello dell'attention (AIM_ATTN):
 *   softmax  : (logsumexp, +)  -> pesi exp(s - lse)            [standard]
 *   max      : (max, +)        -> tutto il peso sull'argmax     [tropicale]
 *   topk=K   : softmax ristretta ai K punteggi piu' alti        [ibrido]
 *   h=X      : dequantizzazione di Maslov, (+)_h = h*log(sum e^(s/h)):
 *              h=1 softmax, h->0 max                            [famiglia] */
static int attn_mode = -1, attn_topk = 0;
static float attn_h = 1.0f;
static void attn_mode_init(void)
{
    if (attn_mode >= 0) return;
    const char *e = getenv("AIM_ATTN");
    attn_mode = 0;
    if (e && !strcmp(e, "max")) attn_mode = 1;
    else if (e && !strncmp(e, "topk=", 5)) { attn_mode = 2; attn_topk = atoi(e + 5); if (attn_topk < 1) attn_topk = 1; }
    else if (e && !strncmp(e, "h=", 2)) { attn_mode = 3; attn_h = (float)atof(e + 2); if (attn_h <= 0) attn_h = 1.0f; }
}


#if defined(__AVX2__)
#include <immintrin.h>
/* exp vettoriale (errore relativo ~1e-6): 2^n * p(r), x = n ln2 + r */
static inline __m256 exp256_ps(__m256 x)
{
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.0f)), _mm256_set1_ps(88.0f));
    __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693145751953125f), x);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(1.428606765330187e-06f), r);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
    __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}
static inline float hsum256(__m256 v)
{
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}
/* 4 prodotti scalari q_g . k (g = 4 query dello stesso kv-head), n multiplo di 8 */
static inline void dot4(const float *q0, const float *q1, const float *q2, const float *q3,
                        const float *k, int n, float *out)
{
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    for (int i = 0; i < n; i += 8) {
        __m256 kv = _mm256_loadu_ps(k + i);
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q0 + i), kv, a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q1 + i), kv, a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(q2 + i), kv, a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(q3 + i), kv, a3);
    }
    out[0] = hsum256(a0); out[1] = hsum256(a1); out[2] = hsum256(a2); out[3] = hsum256(a3);
}
static inline void softmax_inplace(float *sc, int n, float mx, float thr, float *sum_out)
{
    __m256 vmx = _mm256_set1_ps(mx), vthr = _mm256_set1_ps(thr), vsum = _mm256_setzero_ps();
    int t = 0;
    for (; t + 8 <= n; t += 8) {
        __m256 s = _mm256_loadu_ps(sc + t);
        __m256 e = exp256_ps(_mm256_sub_ps(s, vmx));
        e = _mm256_and_ps(e, _mm256_cmp_ps(s, vthr, _CMP_GE_OQ));
        _mm256_storeu_ps(sc + t, e); vsum = _mm256_add_ps(vsum, e);
    }
    float sum = hsum256(vsum);
    for (; t < n; t++) { float e = sc[t] >= thr ? expf(sc[t] - mx) : 0.0f; sc[t] = e; sum += e; }
    *sum_out = sum;
}
static inline void axpy4(float *o0, float *o1, float *o2, float *o3, const float *p, const float *v, int n)
{
    __m256 p0 = _mm256_set1_ps(p[0]), p1 = _mm256_set1_ps(p[1]), p2 = _mm256_set1_ps(p[2]), p3 = _mm256_set1_ps(p[3]);
    for (int i = 0; i < n; i += 8) {
        __m256 vv = _mm256_loadu_ps(v + i);
        _mm256_storeu_ps(o0 + i, _mm256_fmadd_ps(p0, vv, _mm256_loadu_ps(o0 + i)));
        _mm256_storeu_ps(o1 + i, _mm256_fmadd_ps(p1, vv, _mm256_loadu_ps(o1 + i)));
        _mm256_storeu_ps(o2 + i, _mm256_fmadd_ps(p2, vv, _mm256_loadu_ps(o2 + i)));
        _mm256_storeu_ps(o3 + i, _mm256_fmadd_ps(p3, vv, _mm256_loadu_ps(o3 + i)));
    }
}
#else
static inline void dot4(const float *q0, const float *q1, const float *q2, const float *q3, const float *k, int n, float *out)
{
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int i = 0; i < n; i++) { a0 += q0[i] * k[i]; a1 += q1[i] * k[i]; a2 += q2[i] * k[i]; a3 += q3[i] * k[i]; }
    out[0] = a0; out[1] = a1; out[2] = a2; out[3] = a3;
}
static inline void softmax_inplace(float *sc, int n, float mx, float thr, float *sum_out)
{
    float sum = 0;
    for (int t = 0; t < n; t++) { float e = sc[t] >= thr ? expf(sc[t] - mx) : 0.0f; sc[t] = e; sum += e; }
    *sum_out = sum;
}
static inline void axpy4(float *o0, float *o1, float *o2, float *o3, const float *p, const float *v, int n)
{
    for (int i = 0; i < n; i++) { o0[i] += p[0] * v[i]; o1[i] += p[1] * v[i]; o2[i] += p[2] * v[i]; o3[i] += p[3] * v[i]; }
}
#endif

/* Attention in tre fasi sul pool (GQA: le `group` query di un kv-head
 * leggono ogni riga K/V una volta sola; le posizioni sono divise in chunk
 * per avere abbastanza item paralleli anche con un solo token):
 *   1. punteggi     item = (b, kv-head, chunk)
 *   2. pesi         item = (b, head): semianello scelto, exp vettoriale
 *   3. somma pesata item = (b, kv-head, chunk) -> parziali; poi riduzione   */
static void attn_task_fn(void *ctx, int tid, int nth)
{
    (void)tid; (void)nth;
    attn_task *a = ctx;
    const int HD = a->HD, NH = a->NH, NKV = NH / a->group, G = a->group;
    const int max_ctx = a->c->max_ctx, nchunk = a->nchunk, chunk = a->chunk, B = a->B;
    const int n1 = B * NKV * nchunk;
    const float *qs[8]; float *os[8];
    float tmp[8];

    /* fase 1: punteggi */
    for (int it; (it = aim_pool_next(&a->next1, 1)) < n1;) {
        const int ch = it % nchunk, kh = (it / nchunk) % NKV, b = it / (nchunk * NKV);
        const int pos = a->pos0 + b, t0 = ch * chunk, t1 = t0 + chunk <= pos + 1 ? t0 + chunk : pos + 1;
        if (t0 > pos) continue;
        for (int g = 0; g < G; g++) qs[g] = a->q + (size_t)b * a->ldq + (kh * G + g) * HD;
        for (int t = t0; t < t1; t++) {
            const float *kt = a->kbase + ((size_t)kh * max_ctx + t) * HD;
            for (int g = 0; g < G; g += 4) {
                dot4(qs[g], qs[g + 1 < G ? g + 1 : g], qs[g + 2 < G ? g + 2 : g], qs[g + 3 < G ? g + 3 : g], kt, HD, tmp);
                for (int j = 0; j < 4 && g + j < G; j++) {
                    float sv = tmp[j] * a->scale;
                    if (attn_mode == 3) sv /= attn_h;
                    a->sc[((size_t)b * NH + kh * G + g + j) * max_ctx + t] = sv;
                }
            }
        }
    }
    aim_pool_barrier();

    /* fase 2: pesi */
    const int n2 = B * NH;
    for (int it; (it = aim_pool_next(&a->next2, 1)) < n2;) {
        const int b = it / NH, pos = a->pos0 + b, n = pos + 1;
        float *sc = a->sc + (size_t)it * max_ctx;
        float mx = sc[0]; int amax = 0;
        for (int t = 1; t < n; t++) if (sc[t] > mx) { mx = sc[t]; amax = t; }
        if (attn_mode == 1) {                        /* tropicale: one-hot sull'argmax */
            for (int t = 0; t < n; t++) sc[t] = 0.0f;
            sc[amax] = 1.0f;
            continue;
        }
        float thr = -1e30f;
        if (attn_mode == 2 && n > attn_topk) {       /* soglia = K-esimo punteggio */
            float top[64]; int K = attn_topk > 64 ? 64 : attn_topk, m = 0;
            for (int t = 0; t < n; t++) {
                float sv = sc[t];
                if (m < K) { int j = m++; while (j > 0 && top[j - 1] < sv) { top[j] = top[j - 1]; j--; } top[j] = sv; }
                else if (sv > top[K - 1]) { int j = K - 1; while (j > 0 && top[j - 1] < sv) { top[j] = top[j - 1]; j--; } top[j] = sv; }
            }
            thr = top[K - 1];
        }
        float sum; softmax_inplace(sc, n, mx, thr, &sum);
        float inv = 1.0f / sum;
        for (int t = 0; t < n; t++) sc[t] *= inv;
    }
    aim_pool_barrier();

    /* fase 3: somme pesate parziali per chunk */
    for (int it; (it = aim_pool_next(&a->next3, 1)) < n1;) {
        const int ch = it % nchunk, kh = (it / nchunk) % NKV, b = it / (nchunk * NKV);
        const int pos = a->pos0 + b, t0 = ch * chunk, t1 = t0 + chunk <= pos + 1 ? t0 + chunk : pos + 1;
        for (int g = 0; g < G; g++) {
            os[g] = a->part + (((size_t)b * NH + kh * G + g) * nchunk + ch) * HD;
            memset(os[g], 0, HD * sizeof(float));
        }
        if (t0 > pos) continue;
        for (int t = t0; t < t1; t++) {
            int any = 0;
            for (int g = 0; g < G; g++) { tmp[g] = a->sc[((size_t)b * NH + kh * G + g) * max_ctx + t]; any |= tmp[g] != 0.0f; }
            if (!any) continue;
            const float *vt = a->vbase + ((size_t)kh * max_ctx + t) * HD;
            for (int g = 0; g < G; g += 4)
                axpy4(os[g], os[g + 1 < G ? g + 1 : g], os[g + 2 < G ? g + 2 : g], os[g + 3 < G ? g + 3 : g], tmp + g, vt, HD);
        }
    }
    aim_pool_barrier();

    /* fase 4: riduzione dei chunk (item = (b, head)) */
    for (int it = tid; it < n2; it += nth) {
        const int b = it / NH, h = it % NH;
        float *o = a->out + (size_t)b * a->ldo + h * HD;
        const float *pp = a->part + ((size_t)b * NH + h) * nchunk * HD;
        memcpy(o, pp, HD * sizeof(float));
        for (int ch = 1; ch < nchunk; ch++)
            for (int i = 0; i < HD; i++) o[i] += pp[(size_t)ch * HD + i];
    }
}

static attn_task attn_make(aim_ctx *c, const float *q, int ldq, float *out, int ldo, float *sc, float *part,
                           const float *kbase, const float *vbase, int pos0, int B, int kvd, int HD, int NH, int group, float scale)
{
    attn_task a;
    memset(&a, 0, sizeof a);
    a.c = c; a.q = q; a.ldq = ldq; a.out = out; a.ldo = ldo; a.sc = sc; a.part = part;
    a.kbase = kbase; a.vbase = vbase; a.pos0 = pos0; a.B = B; a.kvd = kvd; a.HD = HD; a.NH = NH; a.group = group; a.scale = scale;
    a.chunk = ATTN_CHUNK;
    a.nchunk = (pos0 + B + ATTN_CHUNK - 1) / ATTN_CHUNK;
    return a;
}

int aim_argmax(const float *v, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

const float *aim_forward(aim_ctx *c, int token)
{
    attn_mode_init();
    const aim_model *m = c->m;
    const int H = m->hidden, HD = m->head_dim, NH = m->n_heads, NKV = m->n_kv;
    const int kvd = NKV * HD, half = HD / 2, pos = c->pos, I = m->inter;
    const int group = NH / NKV;
    const float att_scale = 1.0f / sqrtf((float)HD);
    float *x = c->x, *xb = c->xb;

    /* embedding int8 -> float */
    const int8_t *e = m->emb_q + (size_t)token * H;
    float es = m->emb_s[token];
    for (int i = 0; i < H; i++) x[i] = e[i] * es;

    for (int l = 0; l < m->n_layers; l++) {
        const aim_layer *L = &m->layers[l];
        double t0 = aim_now_sec();

        rmsnorm(xb, x, L->ln1, H, m->eps);
        double tk = aim_now_sec();
        aim_t3_gemv_simd(&L->qkv, xb, c->qkv);
        c->t_kind[0] += aim_now_sec() - tk;
        float *q = c->qkv, *k = c->qkv + H, *v = k + kvd;
        c->t_gemv += aim_now_sec() - t0; t0 = aim_now_sec();

        const float *cs = c->rope_cos + (size_t)pos * half, *sn = c->rope_sin + (size_t)pos * half;
        rope(q, NH, HD, cs, sn);
        rope(k, NKV, HD, cs, sn);
        for (int kh = 0; kh < NKV; kh++) {      /* cache [layer][kv-head][pos][HD]: K di un head contigua */
            memcpy(c->k_cache + (((size_t)l * NKV + kh) * c->max_ctx + pos) * HD, k + kh * HD, HD * sizeof(float));
            memcpy(c->v_cache + (((size_t)l * NKV + kh) * c->max_ctx + pos) * HD, v + kh * HD, HD * sizeof(float));
        }

        attn_task at = attn_make(c, q, 0, c->attn, 0, c->att, c->part,
                                 c->k_cache + (size_t)l * c->max_ctx * kvd, c->v_cache + (size_t)l * c->max_ctx * kvd,
                                 pos, 1, kvd, HD, NH, group, att_scale);
        aim_pool_run(attn_task_fn, &at);
        c->t_attn += aim_now_sec() - t0; t0 = aim_now_sec();

        rmsnorm(xb, c->attn, L->attn_sub, H, m->eps);      /* attn_sub_norm */
        tk = aim_now_sec(); aim_t3_gemv_simd(&L->o, xb, c->attn); c->t_kind[1] += aim_now_sec() - tk;
        for (int i = 0; i < H; i++) x[i] += c->attn[i];

        rmsnorm(xb, x, L->ln2, H, m->eps);
        tk = aim_now_sec(); aim_t3_gemv_simd(&L->gate_up, xb, c->gu); c->t_kind[2] += aim_now_sec() - tk;
        float *g = c->gu, *u = c->gu + I;
        for (int i = 0; i < I; i++) { float r = g[i] > 0 ? g[i] : 0; c->h[i] = r * r * u[i]; }   /* relu^2 */
        rmsnorm(c->h, c->h, L->ffn_sub, I, m->eps);          /* ffn_sub_norm */
        tk = aim_now_sec(); aim_t3_gemv_simd(&L->down, c->h, xb); c->t_kind[3] += aim_now_sec() - tk;
        for (int i = 0; i < H; i++) x[i] += xb[i];
        c->t_gemv += aim_now_sec() - t0;
        if (getenv("AIM_DUMP") && (pos == 0 || l == m->n_layers - 1 || l == 0)) {
            double sa = 0; for (int i = 0; i < H; i++) sa += fabs(x[i]);
            fprintf(stderr, "layer %2d: x[0..3] = %9.4f %9.4f %9.4f %9.4f  sum|x| = %.3f\n", l, x[0], x[1], x[2], x[3], sa);
        }
    }

    double t0 = aim_now_sec();
    rmsnorm(xb, x, m->final_norm, H, m->eps);
    aim_gemv_i8(m->emb_q, m->emb_s, m->vocab, H, xb, c->logits);
    c->t_head += aim_now_sec() - t0;
    c->pos++;
    return c->logits;
}

/* ---------------------------------------------------------------- batch */
const float *aim_forward_batch(aim_ctx *c, const int *tokens, int B)
{
    return aim_forward_batch_logits(c, tokens, B, NULL);
}

/* come aim_forward_batch; se all_logits != NULL scrive i logit di ogni token (B x vocab) */
const float *aim_forward_batch_logits(aim_ctx *c, const int *tokens, int B, float *all_logits)
{
    attn_mode_init();
    const aim_model *m = c->m;
    const int H = m->hidden, HD = m->head_dim, NH = m->n_heads, NKV = m->n_kv;
    const int kvd = NKV * HD, half = HD / 2, pos0 = c->pos, I = m->inter;
    const int group = NH / NKV, ldq = H + 2 * kvd;
    const float att_scale = 1.0f / sqrtf((float)HD);

    float *X   = malloc((size_t)B * H * sizeof(float));
    float *XB  = malloc((size_t)B * H * sizeof(float));
    float *QKV = malloc((size_t)B * ldq * sizeof(float));
    float *ATT = malloc((size_t)B * H * sizeof(float));
    float *GU  = malloc((size_t)B * 2 * I * sizeof(float));
    float *HH  = malloc((size_t)B * I * sizeof(float));
    float *SC  = malloc((size_t)B * NH * c->max_ctx * sizeof(float));
    int nchunk_max = (c->max_ctx + ATTN_CHUNK - 1) / ATTN_CHUNK;
    float *PART = malloc((size_t)B * NH * nchunk_max * HD * sizeof(float));

    for (int b = 0; b < B; b++) {
        const int8_t *e = m->emb_q + (size_t)tokens[b] * H;
        float es = m->emb_s[tokens[b]];
        for (int i = 0; i < H; i++) X[(size_t)b * H + i] = e[i] * es;
    }

    for (int l = 0; l < m->n_layers; l++) {
        const aim_layer *L = &m->layers[l];
        double t0 = aim_now_sec();
        for (int b = 0; b < B; b++) rmsnorm(XB + (size_t)b * H, X + (size_t)b * H, L->ln1, H, m->eps);
        double tk = aim_now_sec(); aim_t3_gemm_simd(&L->qkv, XB, H, B, QKV, ldq); c->t_kind[0] += aim_now_sec() - tk;
        c->t_gemv += aim_now_sec() - t0; t0 = aim_now_sec();

        for (int b = 0; b < B; b++) {
            int pos = pos0 + b;
            float *q = QKV + (size_t)b * ldq, *k = q + H, *v = k + kvd;
            const float *cs = c->rope_cos + (size_t)pos * half, *sn = c->rope_sin + (size_t)pos * half;
            rope(q, NH, HD, cs, sn);
            rope(k, NKV, HD, cs, sn);
            for (int kh = 0; kh < NKV; kh++) {
                memcpy(c->k_cache + (((size_t)l * NKV + kh) * c->max_ctx + pos) * HD, k + kh * HD, HD * sizeof(float));
                memcpy(c->v_cache + (((size_t)l * NKV + kh) * c->max_ctx + pos) * HD, v + kh * HD, HD * sizeof(float));
            }
        }
        attn_task at = attn_make(c, QKV, ldq, ATT, H, SC, PART,
                                 c->k_cache + (size_t)l * c->max_ctx * kvd, c->v_cache + (size_t)l * c->max_ctx * kvd,
                                 pos0, B, kvd, HD, NH, group, att_scale);
        aim_pool_run(attn_task_fn, &at);
        c->t_attn += aim_now_sec() - t0; t0 = aim_now_sec();

        for (int b = 0; b < B; b++) rmsnorm(XB + (size_t)b * H, ATT + (size_t)b * H, L->attn_sub, H, m->eps);
        tk = aim_now_sec(); aim_t3_gemm_simd(&L->o, XB, H, B, ATT, H); c->t_kind[1] += aim_now_sec() - tk;
        for (size_t i = 0; i < (size_t)B * H; i++) X[i] += ATT[i];

        for (int b = 0; b < B; b++) rmsnorm(XB + (size_t)b * H, X + (size_t)b * H, L->ln2, H, m->eps);
        tk = aim_now_sec(); aim_t3_gemm_simd(&L->gate_up, XB, H, B, GU, 2 * I); c->t_kind[2] += aim_now_sec() - tk;
        for (int b = 0; b < B; b++) {
            const float *g = GU + (size_t)b * 2 * I, *u = g + I;
            float *h = HH + (size_t)b * I;
            for (int i = 0; i < I; i++) { float r = g[i] > 0 ? g[i] : 0; h[i] = r * r * u[i]; }
            rmsnorm(h, h, L->ffn_sub, I, m->eps);
        }
        tk = aim_now_sec(); aim_t3_gemm_simd(&L->down, HH, I, B, XB, H); c->t_kind[3] += aim_now_sec() - tk;
        for (size_t i = 0; i < (size_t)B * H; i++) X[i] += XB[i];
        c->t_gemv += aim_now_sec() - t0;
    }

    double t0 = aim_now_sec();
    if (all_logits) {
        for (int b = 0; b < B; b++) {
            rmsnorm(c->xb, X + (size_t)b * H, m->final_norm, H, m->eps);
            aim_gemv_i8(m->emb_q, m->emb_s, m->vocab, H, c->xb, all_logits + (size_t)b * m->vocab);
        }
        memcpy(c->logits, all_logits + (size_t)(B - 1) * m->vocab, (size_t)m->vocab * sizeof(float));
    } else {
        rmsnorm(c->xb, X + (size_t)(B - 1) * H, m->final_norm, H, m->eps);
        aim_gemv_i8(m->emb_q, m->emb_s, m->vocab, H, c->xb, c->logits);
    }
    c->t_head += aim_now_sec() - t0;
    c->pos += B;

    free(X); free(XB); free(QKV); free(ATT); free(GU); free(HH); free(SC); free(PART);
    return c->logits;
}

double aim_perplexity(aim_ctx *c, const int *ids, int n, int batch)
{
    const int V = c->m->vocab;
    float *L = malloc((size_t)batch * V * sizeof(float));
    double nll = 0; int cnt = 0;
    for (int i = 0; i < n; ) {
        int nb = n - i < batch ? n - i : batch;
        aim_forward_batch_logits(c, ids + i, nb, L);
        for (int b = 0; b < nb; b++) {
            int next = i + b + 1;
            if (next >= n) break;
            const float *lg = L + (size_t)b * V;
            float mx = lg[0]; for (int v = 1; v < V; v++) if (lg[v] > mx) mx = lg[v];
            double se = 0; for (int v = 0; v < V; v++) se += exp((double)lg[v] - mx);
            nll += -((double)lg[ids[next]] - mx - log(se));
            cnt++;
        }
        i += nb;
    }
    free(L);
    return cnt ? exp(nll / cnt) : 0;
}
