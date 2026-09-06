#include "../src/model.c"
#include <stdio.h>
int main(void)
{
#if defined(__AVX2__)
    double worst = 0; float xs[8], wx = 0, wo = 0;
    for (float x = -87.0f; x < 88.0f; x += 0.0173f) {
        for (int i = 0; i < 8; i++) xs[i] = x + i * 0.001f;
        float out[8]; _mm256_storeu_ps(out, exp256_ps(_mm256_loadu_ps(xs)));
        for (int i = 0; i < 8; i++) { double e = fabs((double)out[i] / exp((double)xs[i]) - 1.0); if (e > worst) { worst = e; wx = xs[i]; wo = out[i]; } }
    }
    printf("exp256_ps: errore relativo massimo %.3e a x=%.5f (got %.8e, exp %.8e)\n", worst, wx, wo, exp((double)wx));
    double w2 = 0;
    for (float x = -20.0f; x < 20.0f; x += 0.0173f) {
        for (int i = 0; i < 8; i++) xs[i] = x + i * 0.001f;
        float out[8]; _mm256_storeu_ps(out, exp256_ps(_mm256_loadu_ps(xs)));
        for (int i = 0; i < 8; i++) { double e = fabs((double)out[i] / exp((double)xs[i]) - 1.0); if (e > w2) w2 = e; }
    }
    printf("su [-20,20]: %.3e\n", w2);
#endif
    return 0;
}
