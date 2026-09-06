#include "aim.h"
#include <stdio.h>
#include <stdlib.h>
static void empty_task(void *c, int tid, int nth) { (void)c; (void)tid; (void)nth; }
static void barrier_task(void *c, int tid, int nth) { (void)c; (void)tid; (void)nth; aim_pool_barrier(); }
int main(void)
{
    aim_pool_init(0);
    int N = 5000;
    aim_pool_run(empty_task, NULL);
    double t0 = aim_now_sec();
    for (int i = 0; i < N; i++) aim_pool_run(empty_task, NULL);
    double t1 = (aim_now_sec() - t0) / N;
    t0 = aim_now_sec();
    for (int i = 0; i < N; i++) aim_pool_run(barrier_task, NULL);
    double t2 = (aim_now_sec() - t0) / N;
    printf("pool %2d thread: dispatch vuoto %.1f us | dispatch+barriera %.1f us\n", aim_pool_size(), t1 * 1e6, t2 * 1e6);
    aim_pool_shutdown();
    return 0;
}
