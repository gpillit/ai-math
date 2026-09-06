/* aim_model.h — inferenza di un modello BitNet b1.58 (architettura Llama con
 * sub-norm) sopra la matrice ternaria in base 3.  Formato file .aim scritto
 * da tools/convert.py.  Tutto il modello sta in RAM in un unico blob. */
#ifndef AIM_MODEL_H
#define AIM_MODEL_H

#include "aim.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const float *ln1, *attn_sub, *ln2, *ffn_sub;
    aim_t3_tiled qkv, o, gate_up, down;     /* puntano nel blob, non posseggono */
} aim_layer;

typedef struct {
    int n_layers, hidden, inter, n_heads, n_kv, head_dim, vocab, max_pos;
    float eps, rope_theta;
    const int8_t *emb_q;   /* [vocab][hidden] int8 per riga (anche lm_head, tied) */
    const float  *emb_s;   /* [vocab] */
    const float  *final_norm;
    aim_layer *layers;
    uint8_t *blob;
    size_t blob_size;
    size_t ternary_bytes;  /* byte letti per token dai layer ternari */
} aim_model;

typedef struct {
    const aim_model *m;
    int max_ctx, pos;
    float *k_cache, *v_cache;   /* [layer][pos][n_kv*head_dim] */
    float *x, *xb, *qkv, *att, *attn, *gu, *h, *logits;
    float *rope_cos, *rope_sin; /* [max_ctx][head_dim/2] */
    double t_gemv, t_attn, t_head;   /* profiling cumulativo (secondi) */
    double t_kind[4];                /* qkv, o, gate_up, down */
} aim_ctx;

int   aim_model_load(const char *path, aim_model *m);
void  aim_model_free(aim_model *m);
int   aim_ctx_init(aim_ctx *c, const aim_model *m, int max_ctx);
void  aim_ctx_free(aim_ctx *c);
/* un passo: consuma `token` alla posizione corrente, ritorna i logit [vocab] */
const float *aim_forward(aim_ctx *c, int token);
int   aim_argmax(const float *v, int n);

/* GEMV int8 (pesi int8 per riga con scala, attivazioni int8 per-tensore) */
void aim_gemv_i8(const int8_t *W, const float *ws, int rows, int cols,
                 const float *x, float *y);

#ifdef __cplusplus
}
#endif
#endif
