#include "aim.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fails++; } else printf("ok:   %s\n", msg); } while (0)

int main(void)
{
    aim_pool_init(0);
    aim_t3_init();

    /* 1. la codifica base-3 è biunivoca sui 243 codici */
    int seen[AIM_T3_CODES] = {0};
    for (int b = 0; b < AIM_T3_CODES; b++) {
        int v = 0, p = 1;
        for (int i = 0; i < AIM_T3_GROUP; i++) { v += (aim_t3_decode[b][i] + 1) * p; p *= 3; }
        seen[v]++;
    }
    int bij = 1;
    for (int b = 0; b < AIM_T3_CODES; b++) if (seen[b] != 1) bij = 0;
    CHECK(bij, "decode base-3 e' una biiezione su 243 codici");

    /* 2. quantize -> dequantize e' idempotente sui valori ternari */
    int rows = 37, cols = 103;  /* volutamente non multipli di 5 */
    float *W = malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows * cols; i++) W[i] = (float)((i * 7919) % 3 - 1) * 0.25f;
    aim_t3_mat m;
    CHECK(aim_t3_quantize(W, rows, cols, &m) == 0, "quantize alloca");
    float *Wd = malloc(rows * cols * sizeof(float));
    aim_t3_dequantize(&m, Wd);
    /* la scala e' mean|w| per riga: confrontiamo il pattern ternario, non il valore */
    int same = 1;
    for (int r = 0; r < rows && same; r++)
        for (int c = 0; c < cols; c++) {
            float t = W[r * cols + c] / 0.25f, td = Wd[r * cols + c] / m.scale[r];
            if (fabsf(t - td) > 1e-5f) { same = 0; break; }
        }
    CHECK(same, "dequantize(quantize(W)) == W a meno della scala per riga");

    /* 3. gemv lut == gemv ref == gemv fp32 sulla matrice dequantizzata */
    float *x = malloc(cols * sizeof(float));
    aim_fill_gaussian(x, cols, 42);
    float *y0 = malloc(rows * sizeof(float)), *y1 = malloc(rows * sizeof(float)), *y2 = malloc(rows * sizeof(float));
    aim_gemv_f32(Wd, x, y0, rows, cols);
    aim_t3_gemv_ref(&m, x, y1);
    aim_t3_gemv_lut(&m, x, y2);
    CHECK(aim_rel_err(y0, y1, rows) < 1e-5, "t3_gemv_ref == fp32");
    CHECK(aim_rel_err(y0, y2, rows) < 1e-5, "t3_gemv_lut == fp32");

    /* 4. semianelli: reale coincide con fp32, tropicale = max(w+x) */
    float *y3 = malloc(rows * sizeof(float));
    aim_gemv_sr(&AIM_SR_REAL, Wd, x, y3, rows, cols);
    CHECK(aim_rel_err(y0, y3, rows) < 1e-5, "semiring REAL == fp32");
    aim_gemv_sr(&AIM_SR_TROPICAL, Wd, x, y3, rows, cols);
    int trop_ok = 1;
    for (int r = 0; r < rows && trop_ok; r++) {
        float mx = -INFINITY;
        for (int c = 0; c < cols; c++) { float v = Wd[r * cols + c] + x[c]; if (v > mx) mx = v; }
        if (fabsf(mx - y3[r]) > 1e-6f) trop_ok = 0;
    }
    CHECK(trop_ok, "semiring TROPICAL == max_c (w+x)");

    /* 5. fattorizzazione: (b*19)>>9 == b/27 per ogni codice */
    int div_ok = 1;
    for (unsigned b = 0; b < AIM_T3_CODES; b++) if (((b * 19u) >> 9) != b / 27) div_ok = 0;
    CHECK(div_ok, "(b*19)>>9 == b/27 per b < 243");

    /* 6. varianti veloci == fp32 (blocked, factored esatte; simd con x int8) */
    aim_t3_gemv_lut_blocked(&m, x, y2);
    CHECK(aim_rel_err(y0, y2, rows) < 1e-5, "t3_gemv_lut_blocked == fp32");
    aim_t3_gemv_lut_factored(&m, x, y2);
    CHECK(aim_rel_err(y0, y2, rows) < 1e-5, "t3_gemv_lut_factored == fp32");
    aim_t3_tiled tl;
    CHECK(aim_t3_tile(&m, &tl) == 0, "tile alloca");
    /* x intere in [-127,127] -> quantizzazione int8 esatta -> risultato esatto */
    float *xi = malloc(cols * sizeof(float));
    for (int c = 0; c < cols; c++) xi[c] = (float)((c * 37) % 255 - 127);
    aim_gemv_f32(Wd, xi, y0, rows, cols);
    aim_t3_gemv_simd(&tl, xi, y2);
    CHECK(aim_rel_err(y0, y2, rows) < 1e-6, "t3_gemv_simd esatto su x int8");
    aim_gemv_f32(Wd, x, y0, rows, cols);
    aim_t3_gemv_simd(&tl, x, y2);
    double e_simd = aim_rel_err(y0, y2, rows);
    printf("      errore simd con x gaussiana (quantizzazione int8 attivazioni): %.2e\n", e_simd);
    CHECK(e_simd < 2e-2, "t3_gemv_simd ~ fp32 su x gaussiana (< 2%)");
    aim_t3_tiled_free(&tl);
    free(xi);

    /* 7. bit per peso */
    double bpw = 8.0 * m.rows * m.bytes_per_row / ((double)rows * cols);
    printf("      bit/peso effettivi (senza scale): %.3f\n", bpw);
    CHECK(bpw < 1.7, "packing < 1.7 bit/peso");

    aim_t3_free(&m);
    free(W); free(Wd); free(x); free(y0); free(y1); free(y2); free(y3);
    printf(fails ? "\n%d TEST FALLITI\n" : "\nTUTTI I TEST PASSANO\n", fails);
    return fails ? 1 : 0;
}
