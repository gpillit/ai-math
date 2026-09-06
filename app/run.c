/* run: genera token con un modello .aim
 *   build/run models/x.aim --ids 128000,9906,... [-n 64] [--eos 128001,128009] [--ctx 512]
 * stdout: gli id generati, uno per riga (flush immediato).  stderr: statistiche. */
#include "aim_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_ids(const char *s, int *out, int max)
{
    int n = 0;
    while (*s && n < max) {
        char *e; long v = strtol(s, &e, 10);
        if (e == s) break;
        out[n++] = (int)v; s = e;
        while (*s == ',' || *s == ' ') s++;
    }
    return n;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "uso: %s model.aim --ids 1,2,3 [-n N] [--eos a,b] [--ctx C]\n", argv[0]); return 1; }
    int ids[4096], n_ids = 0, eos[8], n_eos = 0, n_new = 64, ctx = 1024, check = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--ids") && i + 1 < argc) n_ids = parse_ids(argv[++i], ids, 4096);
        else if (!strcmp(argv[i], "--eos") && i + 1 < argc) n_eos = parse_ids(argv[++i], eos, 8);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--check")) check = 1;
    }
    if (!n_ids && !check) { fprintf(stderr, "nessun token in input\n"); return 1; }

    aim_pool_init(0);   /* AIM_THREADS o numero di CPU */
    aim_model m;
    double t0 = aim_now_sec();
    if (aim_model_load(argv[1], &m)) return 1;
    fprintf(stderr, "thread: %d | modello: %d layer, hidden %d, inter %d, %d/%d head, vocab %d | %.0f MB in RAM "
                    "(%.0f MB ternari = %.2f bit/peso) | caricato in %.2f s\n",
            aim_pool_size(), m.n_layers, m.hidden, m.inter, m.n_heads, m.n_kv, m.vocab, m.blob_size / 1e6,
            m.ternary_bytes / 1e6, 8.0 * m.ternary_bytes / (m.ternary_bytes * 5.0), aim_now_sec() - t0);

    if (check) {   /* confronto con tools/check.py */
        const aim_t3_tiled *t = &m.layers[0].qkv;
        float *x = malloc(t->cols * sizeof(float)), *y = malloc(t->rows * sizeof(float));
        for (int i = 0; i < t->cols; i++) x[i] = (float)((i * 37) % 255 - 127);
        aim_t3_gemv_simd(t, x, y);
        double sum = 0; for (int i = 0; i < t->rows; i++) sum += y[i];
        printf("qkv rows %d cols %d scale[0] %.6g scale[%d] %.6g\n", t->rows, t->cols, t->scale[0], t->rows - 1, t->scale[t->rows - 1]);
        printf("y[0..5] = %.4f %.4f %.4f %.4f %.4f %.4f  sum=%.4f\n", y[0], y[1], y[2], y[3], y[4], y[5], sum);
        printf("emb[0][0..3] = %d %d %d %d  s=%.6g\n", m.emb_q[0], m.emb_q[1], m.emb_q[2], m.emb_q[3], m.emb_s[0]);
        return 0;
    }

    aim_ctx c;
    if (aim_ctx_init(&c, &m, ctx)) { fprintf(stderr, "ctx alloc fallita\n"); return 1; }

    /* prefill: un token alla volta */
    t0 = aim_now_sec();
    const float *logits = NULL;
    const char *eb = getenv("AIM_BATCH");
    int batch = eb ? atoi(eb) : 32;
    if (batch < 1) batch = 1;
    for (int i = 0; i < n_ids && c.pos < ctx; ) {
        int nb = n_ids - i < batch ? n_ids - i : batch;
        if (nb + c.pos > ctx) nb = ctx - c.pos;
        if (nb > 1 && !getenv("AIM_DUMP")) { logits = aim_forward_batch(&c, ids + i, nb); i += nb; continue; }
        logits = aim_forward(&c, ids[i]);
        if (getenv("AIM_DUMP")) {
            int top[5]; float lv[5];
            for (int k = 0; k < 5; k++) { top[k] = -1; lv[k] = -1e30f; }
            for (int v = 0; v < m.vocab; v++)
                for (int k = 0; k < 5; k++)
                    if (logits[v] > lv[k]) { for (int j = 4; j > k; j--) { lv[j] = lv[j-1]; top[j] = top[j-1]; } lv[k] = logits[v]; top[k] = v; break; }
            fprintf(stderr, "pos %d top5: [%d, %d, %d, %d, %d] [%.3f, %.3f, %.3f, %.3f, %.3f]\n", i, top[0], top[1], top[2], top[3], top[4], lv[0], lv[1], lv[2], lv[3], lv[4]);
        }
        i++;
    }
    double t_prefill = aim_now_sec() - t0;

    /* decode greedy */
    t0 = aim_now_sec();
    int generated = 0;
    for (int i = 0; i < n_new && c.pos < ctx; i++) {
        int tok = aim_argmax(logits, m.vocab);
        printf("%d\n", tok); fflush(stdout);
        generated++;
        int stop = 0;
        for (int e = 0; e < n_eos; e++) if (tok == eos[e]) stop = 1;
        if (stop) break;
        logits = aim_forward(&c, tok);
    }
    double t_dec = aim_now_sec() - t0;
    int dec_steps = generated - 1 > 0 ? generated - 1 : 1;

    fprintf(stderr, "prefill: %d token in %.2f s (%.1f tok/s, batch %d)\n", n_ids, t_prefill, n_ids / t_prefill, batch);
    fprintf(stderr, "decode : %d token in %.2f s (%.1f tok/s, %.1f ms/token)\n",
            generated, t_dec, dec_steps / t_dec, 1e3 * t_dec / dec_steps);
    double steps = n_ids + dec_steps;   /* medie su prefill+decode (il prefill a batch le abbassa) */
    fprintf(stderr, "per token: gemv ternari %.1f ms, attention %.1f ms, lm_head %.1f ms\n",
            1e3 * c.t_gemv / steps, 1e3 * c.t_attn / steps, 1e3 * c.t_head / steps);
    fprintf(stderr, "banda pesi ternari nei gemv: %.1f GB/s\n", m.ternary_bytes / (c.t_gemv / steps) / 1e9);

    {
        const char *names[4] = { "qkv", "o", "gate_up", "down" };
        const aim_t3_tiled *ts[4] = { &m.layers[0].qkv, &m.layers[0].o, &m.layers[0].gate_up, &m.layers[0].down };
        for (int k = 0; k < 4; k++) {
            double bytes = (double)ts[k]->rows_pad * ts[k]->G * m.n_layers, tt = c.t_kind[k] / steps;
            fprintf(stderr, "  %-8s %5d x %4d  %5.2f MB/layer  %6.3f ms/token  %5.1f GB/s\n", names[k], ts[k]->rows, ts[k]->cols,
                    bytes / m.n_layers / 1e6, 1e3 * tt, bytes / tt / 1e9);
        }
    }

    aim_ctx_free(&c);
    aim_model_free(&m);
    aim_pool_shutdown();
    return 0;
}
