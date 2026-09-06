/* Thread pool persistente con dispatch e barriere a spin.
 * libgomp (mingw) costa ~50 us per regione parallela e ~150 us con barriere:
 * con 120 GEMV per token e' meta' del tempo di inferenza. Qui: ~1-2 us.
 *
 * - I worker spinnano ~2 ms (i GEMV di un token arrivano uno dietro l'altro)
 *   e poi dormono davvero (WaitOnAddress / condvar): su un chip mobile il
 *   budget di potenza e' condiviso e i core che spinnano a vuoto rallentano
 *   quelli che lavorano.
 * - Ogni thread e' fissato a un CPU logico, in ordine di classe di efficienza
 *   (P-core, poi E-core). Sulle CPU ibride i 2 core LP-E (ultimi indici)
 *   sono esclusi di default: il sistema li parcheggia e un thread finito
 *   li' sopra blocca tutte le barriere (14 thread: 4 s/token; 12: 30 tok/s). */
#include "aim.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#include <sched.h>
#endif
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define SPIN_PAUSE() _mm_pause()
#else
#define SPIN_PAUSE() ((void)0)
#endif

#define SPIN_SECONDS 0.002   /* spin prima di dormire */
#define MAX_CPUS 256

typedef struct { int cpu, cls; } cpu_info;

static struct {
    int n;
    pthread_t *th;
    cpu_info cpus[MAX_CPUS];
    int ncpus;
    atomic_int epoch;     /* incrementato ad ogni dispatch */
    atomic_int done;      /* worker che hanno finito il task corrente */
    atomic_int quit;
    aim_task_fn fn;
    void *ctx;
    atomic_int bar_count, bar_gen;
#if !defined(_WIN32)
    pthread_mutex_t mu; pthread_cond_t cv;
#endif
} P;

/* ---- topologia: CPU logici ordinati per classe di efficienza decrescente ---- */
static int cpu_topology(cpu_info *out)
{
    int n = 0;
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    char *buf = len ? malloc(len) : NULL;
    if (buf && GetLogicalProcessorInformationEx(RelationProcessorCore, (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)buf, &len)) {
        for (DWORD off = 0; off < len;) {
            SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *e = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(buf + off);
            for (WORD g = 0; g < e->Processor.GroupCount; g++) {
                KAFFINITY m = e->Processor.GroupMask[g].Mask;
                for (int b = 0; b < 64 && n < MAX_CPUS; b++)
                    if (m & ((KAFFINITY)1 << b)) { out[n].cpu = b; out[n].cls = e->Processor.EfficiencyClass; n++; }
            }
            off += e->Size;
        }
    }
    free(buf);
    if (!n) { SYSTEM_INFO si; GetSystemInfo(&si); for (DWORD i = 0; i < si.dwNumberOfProcessors && n < MAX_CPUS; i++) { out[n].cpu = (int)i; out[n].cls = 0; n++; } }
#else
    int c = (int)sysconf(_SC_NPROCESSORS_ONLN);
    for (int i = 0; i < c && n < MAX_CPUS; i++) { out[n].cpu = i; out[n].cls = 0; n++; }
#endif
    /* insertion sort: classe decrescente, poi indice crescente */
    for (int i = 1; i < n; i++) {
        cpu_info k = out[i]; int j = i - 1;
        while (j >= 0 && (out[j].cls < k.cls || (out[j].cls == k.cls && out[j].cpu > k.cpu))) { out[j + 1] = out[j]; j--; }
        out[j + 1] = k;
    }
    return n;
}

static int default_threads(void)
{
    int classes = 0, per_class[256] = {0};
    for (int i = 0; i < P.ncpus; i++) { if (!per_class[P.cpus[i].cls]) classes++; per_class[P.cpus[i].cls]++; }
    /* CPU ibrida: Windows mette E-core e LP-E-core nella stessa classe, ma i
     * 2 LP-E (sempre gli ultimi indici) vengono parcheggiati dal sistema e
     * rallentano tutto. Default: tutti meno 2. AIM_THREADS per cambiare. */
    if (classes >= 2 && P.ncpus > 4) return P.ncpus - 2;
    return P.ncpus;
}

static void pin_to(int slot)
{
    if (P.ncpus == 0) return;
    int cpu = P.cpus[slot % P.ncpus].cpu;
#if defined(_WIN32)
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu);
#elif defined(__linux__)
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
#endif
}

/* ---- attesa ---- */
static void wait_epoch_change(int seen)
{
#if defined(_WIN32)
    WaitOnAddress(&P.epoch, &seen, sizeof(int), INFINITE);
#else
    pthread_mutex_lock(&P.mu);
    while (atomic_load(&P.epoch) == seen && !atomic_load(&P.quit)) pthread_cond_wait(&P.cv, &P.mu);
    pthread_mutex_unlock(&P.mu);
#endif
}

static void wake_all(void)
{
#if defined(_WIN32)
    WakeByAddressAll(&P.epoch);
#else
    pthread_mutex_lock(&P.mu); pthread_cond_broadcast(&P.cv); pthread_mutex_unlock(&P.mu);
#endif
}

static void *worker(void *arg)
{
    int tid = (int)(intptr_t)arg, seen = 0;
    pin_to(tid);
    for (;;) {
        double t0 = 0; long spins = 0;
        while (atomic_load_explicit(&P.epoch, memory_order_acquire) == seen) {
            if (atomic_load_explicit(&P.quit, memory_order_relaxed)) return NULL;
            SPIN_PAUSE();
            if ((++spins & 63) == 0) {
                double now = aim_now_sec();
                if (t0 == 0) t0 = now;
                else if (now - t0 > SPIN_SECONDS) { wait_epoch_change(seen); t0 = 0; }
            }
        }
        if (atomic_load(&P.quit)) return NULL;
        seen = atomic_load_explicit(&P.epoch, memory_order_acquire);
        P.fn(P.ctx, tid, P.n);
        atomic_fetch_add_explicit(&P.done, 1, memory_order_release);
    }
}

int aim_pool_size(void) { return P.n > 0 ? P.n : 1; }

void aim_pool_init(int n)
{
    if (P.n > 0) return;
    P.ncpus = cpu_topology(P.cpus);
    if (n <= 0) { const char *e = getenv("AIM_THREADS"); n = e ? atoi(e) : 0; }
    if (n <= 0) n = default_threads();
    if (n < 1) n = 1;
    if (getenv("AIM_VERBOSE")) {
        fprintf(stderr, "cpu topology (%d logici):", P.ncpus);
        for (int i = 0; i < P.ncpus; i++) fprintf(stderr, " %d/c%d", P.cpus[i].cpu, P.cpus[i].cls);
        fprintf(stderr, "\nthread pool: %d\n", n);
    }
    P.n = n;
    atomic_store(&P.epoch, 0); atomic_store(&P.done, 0); atomic_store(&P.quit, 0);
    atomic_store(&P.bar_count, 0); atomic_store(&P.bar_gen, 0);
#if !defined(_WIN32)
    pthread_mutex_init(&P.mu, NULL); pthread_cond_init(&P.cv, NULL);
#endif
    P.th = calloc(n, sizeof(pthread_t));
    if (n > 1) {
        pin_to(0);
        for (int i = 1; i < n; i++) pthread_create(&P.th[i], NULL, worker, (void *)(intptr_t)i);
    }
}

void aim_pool_shutdown(void)
{
    if (P.n <= 0) return;
    atomic_store(&P.quit, 1);
    atomic_fetch_add(&P.epoch, 1);
    wake_all();
    for (int i = 1; i < P.n; i++) pthread_join(P.th[i], NULL);
    free(P.th); P.th = NULL; P.n = 0;
}

void aim_pool_run(aim_task_fn fn, void *ctx)
{
    if (P.n <= 1) { fn(ctx, 0, 1); return; }
    P.fn = fn; P.ctx = ctx;
    atomic_store_explicit(&P.done, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&P.epoch, 1, memory_order_release);
    wake_all();
    fn(ctx, 0, P.n);
    while (atomic_load_explicit(&P.done, memory_order_acquire) < P.n - 1) SPIN_PAUSE();
}

/* barriera sense-reversing per tutti i P.n partecipanti del task corrente */
void aim_pool_barrier(void)
{
    if (P.n <= 1) return;
    int gen = atomic_load_explicit(&P.bar_gen, memory_order_acquire);
    if (atomic_fetch_add_explicit(&P.bar_count, 1, memory_order_acq_rel) == P.n - 1) {
        atomic_store_explicit(&P.bar_count, 0, memory_order_relaxed);
        atomic_store_explicit(&P.bar_gen, gen + 1, memory_order_release);
    } else {
        while (atomic_load_explicit(&P.bar_gen, memory_order_acquire) == gen) SPIN_PAUSE();
    }
}

int aim_pool_next(atomic_int *counter, int chunk) { return atomic_fetch_add_explicit(counter, chunk, memory_order_relaxed); }
