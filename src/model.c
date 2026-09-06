#include "aim_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    free(c->k_cache); free(c->v_cache); free(c->x); free(c->xb); free(c->qkv); free(c->att);
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
    float *sc;                     /* scratch: (b*NH + h) * max_ctx */
    const float *kbase, *vbase;
    int pos0, B, kvd, HD, NH, group; float scale;
    atomic_int next;
} attn_task;

static void attn_task_fn(void *ctx, int tid, int nth)
{
    (void)tid; (void)nth;
    attn_task *a = ctx;
    const int HD = a->HD, kvd = a->kvd, NH = a->NH, max_ctx = a->c->max_ctx;
    const int nitems = a->B * NH;
    for (int it; (it = aim_pool_next(&a->next, 1)) < nitems;) {
        const int b = it / NH, h = it % NH, pos = a->pos0 + b, kh = h / a->group;
        const float *qh = a->q + (size_t)b * a->ldq + h * HD;
        float *sc = a->sc + (size_t)it * max_ctx;
        float mx = -1e30f;
        for (int t = 0; t <= pos; t++) {
            const float *kt = a->kbase + (size_t)t * kvd + kh * HD;
            float s = 0;
            for (int i = 0; i < HD; i++) s += qh[i] * kt[i];
            s *= a->scale;
            sc[t] = s; if (s > mx) mx = s;
        }
        float sum = 0;
        for (int t = 0; t <= pos; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
        float inv = 1.0f / sum;
        float *o = a->out + (size_t)b * a->ldo + h * HD;
        for (int i = 0; i < HD; i++) o[i] = 0;
        for (int t = 0; t <= pos; t++) {
            const float *vt = a->vbase + (size_t)t * kvd + kh * HD;
            float p = sc[t] * inv;
            for (int i = 0; i < HD; i++) o[i] += p * vt[i];
        }
    }
}

int aim_argmax(const float *v, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

const float *aim_forward(aim_ctx *c, int token)
{
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
        float *kc = c->k_cache + ((size_t)l * c->max_ctx + pos) * kvd;
        float *vc = c->v_cache + ((size_t)l * c->max_ctx + pos) * kvd;
        memcpy(kc, k, kvd * sizeof(float));
        memcpy(vc, v, kvd * sizeof(float));

        attn_task at = { c, q, 0, c->attn, 0, c->att,
                         c->k_cache + (size_t)l * c->max_ctx * kvd, c->v_cache + (size_t)l * c->max_ctx * kvd,
                         pos, 1, kvd, HD, NH, group, att_scale, 0 };
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
            memcpy(c->k_cache + ((size_t)l * c->max_ctx + pos) * kvd, k, kvd * sizeof(float));
            memcpy(c->v_cache + ((size_t)l * c->max_ctx + pos) * kvd, v, kvd * sizeof(float));
        }
        attn_task at = { c, QKV, ldq, ATT, H, SC,
                         c->k_cache + (size_t)l * c->max_ctx * kvd, c->v_cache + (size_t)l * c->max_ctx * kvd,
                         pos0, B, kvd, HD, NH, group, att_scale, 0 };
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
    rmsnorm(c->xb, X + (size_t)(B - 1) * H, m->final_norm, H, m->eps);
    aim_gemv_i8(m->emb_q, m->emb_s, m->vocab, H, c->xb, c->logits);
    c->t_head += aim_now_sec() - t0;
    c->pos += B;

    free(X); free(XB); free(QKV); free(ATT); free(GU); free(HH); free(SC);
    return c->logits;
}
