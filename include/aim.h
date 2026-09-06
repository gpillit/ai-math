/* aim.h — AI-Math: strutture algebriche per inferenza LLM memory-bound.
 *
 * Idea guida: su CPU il costo dell'inferenza autoregressiva NON è il calcolo,
 * è la banda di memoria (ogni token deve leggere tutti i pesi dalla RAM).
 * Quindi una "buona" struttura algebrica per i pesi è una che:
 *   1. minimizza i bit per peso letti dalla RAM,
 *   2. rende il prodotto peso*attivazione una lookup, non una moltiplicazione,
 *   3. resta un modulo su un semianello, così attention/MLP restano esprimibili.
 */
#ifndef AIM_H
#define AIM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* 1. Semianello generico (kernel di riferimento, lento ma universale)  */
/* ------------------------------------------------------------------ */
typedef struct {
    float (*add)(float a, float b);   /* operazione ⊕ */
    float (*mul)(float a, float b);   /* operazione ⊗ */
    float zero;                       /* identità di ⊕ */
    const char *name;
} aim_semiring;

extern const aim_semiring AIM_SR_REAL;      /* (+, ×)          — algebra lineare classica */
extern const aim_semiring AIM_SR_TROPICAL;  /* (max, +)        — nessuna moltiplicazione  */
extern const aim_semiring AIM_SR_LOG;       /* (logsumexp, +)  — softmax "nativo"          */

/* y[r] = ⊕_c  W[r,c] ⊗ x[c]     W è rows×cols row-major */
void aim_gemv_sr(const aim_semiring *sr, const float *W, const float *x,
                 float *y, int rows, int cols);

/* Riferimento fp32 (+,×) veloce, per confronti. */
void aim_gemv_f32(const float *W, const float *x, float *y, int rows, int cols);

/* ------------------------------------------------------------------ */
/* 2. Matrice ternaria in base 3: 5 pesi ∈ {-1,0,+1} per byte          */
/* ------------------------------------------------------------------ */
/* Codifica: byte = Σ_{i<5} (t_i + 1) · 3^i, con t_i ∈ {-1,0,1}.
 * 3^5 = 243 ≤ 256, quindi 1.6 bit/peso (contro 2 bit del packing binario).
 * Il prodotto scalare su un gruppo di 5 colonne è una lookup in una
 * tabella T[243] precalcolata dalle attivazioni: T[b] = Σ_i t_i(b)·x_i.
 */
#define AIM_T3_GROUP 5
#define AIM_T3_CODES 243

typedef struct {
    int rows;
    int cols;          /* colonne logiche */
    int cols_pad;      /* colonne arrotondate a multiplo di 5 */
    int bytes_per_row; /* cols_pad / 5 */
    uint8_t *data;     /* rows * bytes_per_row */
    float   *scale;    /* rows: fattore di scala per riga */
} aim_t3_mat;

int  aim_t3_quantize(const float *W, int rows, int cols, aim_t3_mat *out);
void aim_t3_free(aim_t3_mat *m);
void aim_t3_dequantize(const aim_t3_mat *m, float *W);   /* W rows×cols */
size_t aim_t3_bytes(const aim_t3_mat *m);

/* Kernel: y = scale ∘ (T3 · x) */
void aim_t3_gemv_ref(const aim_t3_mat *m, const float *x, float *y); /* decodifica esplicita */
void aim_t3_gemv_lut(const aim_t3_mat *m, const float *x, float *y); /* lookup table        */

/* Tabelle di decodifica (byte -> 5 trit), inizializzate al primo uso. */
extern int8_t aim_t3_decode[AIM_T3_CODES][AIM_T3_GROUP];
void aim_t3_init(void);

/* ------------------------------------------------------------------ */
/* Thread pool persistente (dispatch e barriere a spin, ~1-2 us)       */
/* ------------------------------------------------------------------ */
#include <stdatomic.h>
typedef void (*aim_task_fn)(void *ctx, int tid, int nth);
void aim_pool_init(int n);            /* n<=0: AIM_THREADS o numero di CPU */
void aim_pool_shutdown(void);
int  aim_pool_size(void);
void aim_pool_run(aim_task_fn fn, void *ctx);   /* fn su tutti i thread, attende la fine */
void aim_pool_barrier(void);                    /* dentro un task */
int  aim_pool_next(atomic_int *counter, int chunk);  /* scheduling dinamico */

/* ------------------------------------------------------------------ */
/* Utilità                                                             */
/* ------------------------------------------------------------------ */
double aim_now_sec(void);
void   aim_fill_gaussian(float *p, size_t n, uint64_t seed);
double aim_rel_err(const float *ref, const float *got, int n);



/* ------------------------------------------------------------------ */
/* 3. Fattorizzazione della tabella e kernel SIMD                       */
/* ------------------------------------------------------------------ */
/* Z_3^5 = Z_3^3 x Z_3^2: un codice b in [0,243) si spezza in
 *   hi = b / 27  in [0,9),   lo = b mod 27  in [0,27)
 * e la tabella T[243] diventa Tlo[27] + Thi[9] (36 voci invece di 243).
 * Le due tabelline entrano in un registro SIMD e la lookup diventa vpshufb.
 * La divisione per 27 su byte e' esatta come (b*19)>>9 per ogni b<243. */
#define AIM_T3_LO 27
#define AIM_T3_HI 9

/* variante A: stessa tabella T[243], ma con blocking di cache */
void aim_t3_gemv_lut_blocked(const aim_t3_mat *m, const float *x, float *y);
/* variante B: tabelle fattorizzate 27+9, scalare */
void aim_t3_gemv_lut_factored(const aim_t3_mat *m, const float *x, float *y);

/* Layout a tile per SIMD: 32 righe per tile, per ogni gruppo 32 byte contigui. */
#define AIM_T3_TILE 32
struct aim_code;
typedef struct {
    int rows, cols, cols_pad, G;   /* G = byte per riga */
    int rows_pad;                  /* multiplo di 32 */
    uint8_t *data;                 /* (rows_pad/32) * G * 32 */
    float   *scale;                /* rows_pad */
    const struct aim_code *code;   /* NULL = ternario base 3 nativo; altrimenti codice generico */
} aim_t3_tiled;

int  aim_t3_tile(const aim_t3_mat *m, aim_t3_tiled *t);
void aim_t3_tiled_free(aim_t3_tiled *t);
/* variante C: AVX2, attivazioni int8 per-tensore, tabelle int16 via vpshufb */
void aim_t3_gemv_simd(const aim_t3_tiled *t, const float *x, float *y);

/* ------------------------------------------------------------------ */
/* 4. GEMM per il prefill: B attivazioni sugli stessi pesi              */
/* ------------------------------------------------------------------ */
/* Con la LUT il calcolo per token non si ammortizza. Qui ogni tile viene
 * spacchettato una volta in int8 (colonne interleaved a 4) e moltiplicato
 * per tutti i B token con vpdpbusd (AVX-VNNI) o vpmaddubsw (AVX2).
 * X: B righe di `cols` float (stride ldx); Y: B righe di `rows` (stride ldy). */
void aim_t3_gemm_simd(const aim_t3_tiled *t, const float *X, int ldx, int B, float *Y, int ldy);


/* ------------------------------------------------------------------ */
/* 5. Codici a byte generici: G = insieme di n-uple di interi piccoli   */
/* ------------------------------------------------------------------ */
/* Un codice e' una biiezione tra al piu' 256 configurazioni ammesse di n
 * pesi (valori interi in [-lev/2, lev/2]) e un byte. Esempi:
 *   t5  : Z3^5, 243 codici, 1.60 bit/peso  (base 3, kernel dedicati)
 *   q4  : Z4^4, 256 codici, 2.00 bit/peso  (2 bit per peso)
 *   s36 : 6 pesi ternari con al piu' 3 non nulli, 233 codici, 1.33 bit/peso
 *         (codice enumerativo: sparsita' strutturata 3:6 a lunghezza fissa)
 * La tabella di lookup per gruppo ha ncodes voci: T[b] = sum_i dec[b][i]*x_i. */
#define AIM_CODE_MAXW 8
typedef struct aim_code {
    const char *name;
    int n;                              /* pesi per byte */
    int ncodes;                         /* codici validi (<= 256) */
    int levels;                         /* livelli per peso (3 = ternario) */
    int kmax;                           /* max non nulli per gruppo (n = nessun vincolo) */
    int8_t dec[256][AIM_CODE_MAXW];     /* codice -> valori */
    int16_t *enc;                       /* indice base-`levels` della n-upla -> codice, -1 se non ammessa */
    double bits_per_weight;
} aim_code;

int  aim_code_init(aim_code *c, const char *name);   /* "t5", "q4", "s36", "sNK" (n pesi, <=K non nulli) */
void aim_code_free(aim_code *c);
int  aim_code_encode(const aim_code *c, const int8_t *vals);   /* -1 se non ammessa */

/* Matrice con codice generico: stesso layout a tile di aim_t3_tiled (G = cols_pad/n).
 * Se t->code == NULL la matrice e' t5 nativa e usa i kernel dedicati. */
int  aim_code_quantize(const aim_code *c, const float *W, int rows, int cols, aim_t3_tiled *t);
/* Pota le n-uple con troppi non nulli (toglie i |w| piu' piccoli) e codifica valori ternari gia' dati */
int  aim_code_pack_ternary(const aim_code *c, const int8_t *T, const float *mag, int rows, int cols,
                           const float *row_scale, aim_t3_tiled *t, long *pruned);
void aim_code_gemv_lut(const aim_t3_tiled *t, const float *x, float *y);    /* LUT scalare generica */
void aim_code_unpack_rows(const aim_t3_tiled *t, int r0, int nrows, int8_t *out, int ld);  /* righe -> int8 */

#ifdef __cplusplus
}
#endif
#endif /* AIM_H */
