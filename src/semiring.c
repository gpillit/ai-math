#include "aim.h"
#include <math.h>
#include <float.h>

static float f_add(float a, float b) { return a + b; }
static float f_mul(float a, float b) { return a * b; }
static float f_max(float a, float b) { return a > b ? a : b; }
static float f_lse(float a, float b) {
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    float m = a > b ? a : b;
    return m + logf(expf(a - m) + expf(b - m));
}

const aim_semiring AIM_SR_REAL     = { f_add, f_mul, 0.0f,      "real (+,x)" };
const aim_semiring AIM_SR_TROPICAL = { f_max, f_add, -INFINITY, "tropical (max,+)" };
const aim_semiring AIM_SR_LOG      = { f_lse, f_add, -INFINITY, "log (lse,+)" };

void aim_gemv_sr(const aim_semiring *sr, const float *W, const float *x,
                 float *y, int rows, int cols)
{
    #pragma omp parallel for schedule(dynamic, 64)
    for (int r = 0; r < rows; r++) {
        const float *w = W + (size_t)r * cols;
        float acc = sr->zero;
        for (int c = 0; c < cols; c++)
            acc = sr->add(acc, sr->mul(w[c], x[c]));
        y[r] = acc;
    }
}

void aim_gemv_f32(const float *W, const float *x, float *y, int rows, int cols)
{
    #pragma omp parallel for schedule(dynamic, 64)
    for (int r = 0; r < rows; r++) {
        const float *w = W + (size_t)r * cols;
        float acc = 0.0f;
        for (int c = 0; c < cols; c++) acc += w[c] * x[c];
        y[r] = acc;
    }
}
