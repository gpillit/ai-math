#include "aim.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

int8_t aim_t3_decode[AIM_T3_CODES][AIM_T3_GROUP];
static int t3_ready = 0;

void aim_t3_init(void)
{
    if (t3_ready) return;
    for (int b = 0; b < AIM_T3_CODES; b++) {
        int v = b;
        for (int i = 0; i < AIM_T3_GROUP; i++) {
            aim_t3_decode[b][i] = (int8_t)(v % 3 - 1);
            v /= 3;
        }
    }
    t3_ready = 1;
}

size_t aim_t3_bytes(const aim_t3_mat *m)
{
    return (size_t)m->rows * m->bytes_per_row + (size_t)m->rows * sizeof(float);
}

/* Quantizzazione ternaria per riga (Ternary Weight Networks, Li & Liu 2016):
 *   delta = 0.7 * mean(|w|)          soglia
 *   t     = sign(w) se |w| > delta, altrimenti 0
 *   scale = mean(|w| : |w| > delta)  scala ottima (minimi quadrati) dato t   */
int aim_t3_quantize(const float *W, int rows, int cols, aim_t3_mat *out)
{
    aim_t3_init();
    out->rows = rows;
    out->cols = cols;
    out->cols_pad = (cols + AIM_T3_GROUP - 1) / AIM_T3_GROUP * AIM_T3_GROUP;
    out->bytes_per_row = out->cols_pad / AIM_T3_GROUP;
    out->data  = calloc((size_t)rows * out->bytes_per_row, 1);
    out->scale = malloc((size_t)rows * sizeof(float));
    if (!out->data || !out->scale) return -1;

    static const int pow3[5] = {1, 3, 9, 27, 81};
    for (int r = 0; r < rows; r++) {
        const float *w = W + (size_t)r * cols;
        double s = 0;
        for (int c = 0; c < cols; c++) s += fabsf(w[c]);
        float delta = 0.7f * (float)(s / cols);
        double s_nz = 0; int n_nz = 0;
        for (int c = 0; c < cols; c++) if (fabsf(w[c]) > delta) { s_nz += fabsf(w[c]); n_nz++; }
        float scale = n_nz ? (float)(s_nz / n_nz) : 1.0f;
        out->scale[r] = scale;
        uint8_t *row = out->data + (size_t)r * out->bytes_per_row;
        for (int c = 0; c < cols; c++) {
            int t = w[c] > delta ? 1 : (w[c] < -delta ? -1 : 0);
            row[c / AIM_T3_GROUP] += (uint8_t)((t + 1) * pow3[c % AIM_T3_GROUP]);
        }
    }
    return 0;
}

void aim_t3_free(aim_t3_mat *m)
{
    free(m->data); free(m->scale);
    m->data = NULL; m->scale = NULL;
}

void aim_t3_dequantize(const aim_t3_mat *m, float *W)
{
    for (int r = 0; r < m->rows; r++) {
        const uint8_t *row = m->data + (size_t)r * m->bytes_per_row;
        for (int c = 0; c < m->cols; c++)
            W[(size_t)r * m->cols + c] =
                m->scale[r] * aim_t3_decode[row[c / AIM_T3_GROUP]][c % AIM_T3_GROUP];
    }
}

/* Riferimento: decodifica ogni trit e moltiplica. */
void aim_t3_gemv_ref(const aim_t3_mat *m, const float *x, float *y)
{
    #pragma omp parallel for schedule(dynamic, 64)
    for (int r = 0; r < m->rows; r++) {
        const uint8_t *row = m->data + (size_t)r * m->bytes_per_row;
        float acc = 0.0f;
        for (int c = 0; c < m->cols; c++)
            acc += aim_t3_decode[row[c / AIM_T3_GROUP]][c % AIM_T3_GROUP] * x[c];
        y[r] = m->scale[r] * acc;
    }
}

/* LUT: per ogni gruppo di 5 colonne costruisce T[243] = Σ_i t_i(b)·x_i.
 * Il prodotto riga·x diventa Σ_g T_g[byte_g]: zero moltiplicazioni sui pesi.
 * Costo tabella: 243·5 FMA per gruppo, ammortizzato su tutte le righe.
 * La tabella è costruita in modo incrementale sfruttando la base 3:
 *   T[b] = T[b - d_k·3^k] + (d_k - 1)·x_k   */
void aim_t3_gemv_lut(const aim_t3_mat *m, const float *x, float *y)
{
    const int G = m->bytes_per_row;
    float *T = malloc((size_t)G * AIM_T3_CODES * sizeof(float));

    #pragma omp parallel for
    for (int g = 0; g < G; g++) {
        float xg[AIM_T3_GROUP];
        for (int i = 0; i < AIM_T3_GROUP; i++) {
            int c = g * AIM_T3_GROUP + i;
            xg[i] = c < m->cols ? x[c] : 0.0f;
        }
        float *Tg = T + (size_t)g * AIM_T3_CODES;
        /* costruzione incrementale: partiamo da b=0 (tutti -1) */
        Tg[0] = -(xg[0] + xg[1] + xg[2] + xg[3] + xg[4]);
        int span = 1;
        for (int i = 0; i < AIM_T3_GROUP; i++) {        /* estende cifra i */
            for (int b = 0; b < span; b++) {
                Tg[b + span]     = Tg[b] + xg[i];       /* digit 1 → t=0  */
                Tg[b + 2 * span] = Tg[b] + 2 * xg[i];   /* digit 2 → t=+1 */
            }
            span *= 3;
        }
    }

    #pragma omp parallel for schedule(dynamic, 64)
    for (int r = 0; r < m->rows; r++) {
        const uint8_t *row = m->data + (size_t)r * m->bytes_per_row;
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        int g = 0;
        for (; g + 4 <= G; g += 4) {
            a0 += T[(size_t)(g + 0) * AIM_T3_CODES + row[g + 0]];
            a1 += T[(size_t)(g + 1) * AIM_T3_CODES + row[g + 1]];
            a2 += T[(size_t)(g + 2) * AIM_T3_CODES + row[g + 2]];
            a3 += T[(size_t)(g + 3) * AIM_T3_CODES + row[g + 3]];
        }
        for (; g < G; g++) a0 += T[(size_t)g * AIM_T3_CODES + row[g]];
        y[r] = m->scale[r] * ((a0 + a1) + (a2 + a3));
    }
    free(T);
}
