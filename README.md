# ai-math

Laboratorio in C per **nuove strutture algebriche orientate all'inferenza LLM
memory-bound** (tutto il modello in RAM, calcolo su CPU).

## Il vincolo fisico che guida tutto

Nell'inferenza autoregressiva ogni token deve leggere *tutti* i pesi dalla RAM.
Su una CPU il calcolo è quasi gratis; la banda verso la memoria no.

    tempo per token  ≈  (byte dei pesi) / (GB/s della RAM)

Quindi una struttura algebrica "buona" per i pesi deve:

1. minimizzare i **bit per peso** effettivamente letti;
2. trasformare `peso ⊗ attivazione` in una **lookup**, non in una moltiplicazione;
3. restare un modulo su un semianello, così attention e MLP restano esprimibili.

Esempio numerico: 70 miliardi di parametri a 1.6 bit/peso = 14 GB, che stanno
in RAM. A 36 GB/s effettivi (misurati qui sotto) sono ~2.5 token/s su un
portatile; un 7B (1.4 GB) fa ~25 token/s.

## Struttura 1: modulo ternario in base 3

Pesi in {-1, 0, +1}, **5 pesi per byte** codificati in base 3
(3⁵ = 243 ≤ 256): 1.6 bit/peso contro i 2 del packing binario.

Il prodotto riga·x non moltiplica mai un peso. Per ogni gruppo di 5 colonne si
costruisce una tabella T[243] = Σ tᵢ·xᵢ e la riga diventa una somma di lookup.

La struttura di gruppo Z₃⁵ = Z₃³ × Z₃² dà la fattorizzazione
**T[b] = Tlo[b mod 27] + Thi[b div 27]**: 36 voci invece di 243, che entrano
nei registri SIMD. La divisione per 27 su un byte è esatta come `(b·19) >> 9`,
oppure `mulhi(b, 2432)`. Con AVX2 la lookup è `vpshufb` su 32 righe per volta.

## Cosa c'è

| file | contenuto |
|---|---|
| `include/aim.h` | API: semianello generico, matrice ternaria, layout a tile |
| `src/semiring.c` | GEMV parametrico su (⊕,⊗): reale, tropicale (max,+), log (lse,+) |
| `src/ternary.c` | quantizzazione TWN, packing base 3, kernel LUT di riferimento |
| `src/ternary_fast.c` | kernel: blocking di cache, fattorizzazione 27×9, AVX2 a tile |
| `tests/test.c` | 16 test: biiezione della codifica, kernel == fp32, semianelli, esattezza SIMD |
| `bench/bench.c` | banda effettiva e compressione |

## Build

Windows (MSYS2 ucrt64): `./build.sh` poi `./build/test` e
`./build/bench [rows cols iters]`. Con `AIM_SKIP_F32=1` il bench non alloca
la matrice fp32 (per misurare il regime DRAM su matrici grandi).
Linux: `make`.

## Risultati (Core Ultra 7 155U: 2 P-core + 8 E-core + 2 LP-E, 14 thread)

Matrice 4096×4096, un GEMV = un token su un layer:

| kernel | byte pesi | ms | GB/s eff. | vs fp32 |
|---|---|---|---|---|
| fp32 | 67 MB | 1.56 | 43 | 1× |
| ternario, LUT T[243] scalare | 3.4 MB | 0.49 | 7 | 3.2× |
| ternario, LUT con blocking L1 | 3.4 MB | 0.40 | 8.4 | 3.9× |
| ternario, 27×9 AVX2 a tile | 3.4 MB | **0.185** | 18 | **8.4×** |

Regime DRAM (pesi troppo grandi per la cache L3 da 12 MB):

| matrice | byte pesi | ms | GB/s eff. |
|---|---|---|---|
| 16384×4096 | 13.5 MB | 0.485 | 28 |
| 65536×4096 | 54 MB | 1.42 | **38** |

fp32 in streaming misura 40-43 GB/s su questa macchina: **con i pesi in DRAM
il kernel AVX2 è quindi limitato dalla RAM, non dal calcolo**, che era
l'obiettivo. Sulla matrice piccola (in L3) resta compute-bound: ~35 istruzioni
vettoriali per 32 byte di pesi, 14 delle quali sulla porta shuffle.

Per tipo di core (16384×4096): P-core 11.9 GB/s, E-core 8.6, LP-E 1.5.
Con AVX-512BW la lookup a 27 voci int16 sarebbe un solo `vpermw` e le
istruzioni per gruppo scenderebbero da ~35 a ~8.

Errore di quantizzazione su pesi gaussiani casuali: 0.435 relativo (TWN).
È l'errore del *quantizzatore*, non del kernel: quello è esatto (vedi test).
Nei modelli addestrati ternari (BitNet b1.58) questo errore non esiste.

## Metodo di lavoro

Per ogni idea: (1) assiomi della struttura, (2) kernel in C, (3) misura di
bit/peso, GB/s, errore, (4) verifica in letteratura che non esista già.
Si chiama "nuova" solo dopo il punto 4 e solo se batte le baseline.

Vicini di casa da conoscere (non nuovi): LUT-GEMM, T-MAC, BitNet b1.58 e i
kernel TL1/TL2 di bitnet.cpp, quantizzazione su reticoli (QuIP#, AQLM).
La fattorizzazione 27×9 della tabella base-3 è la parte da verificare in
letteratura prima di chiamarla nuova.

## Prossimi passi candidati

1. Cambio di semianello dentro il modello: layer in (max,+), niente
   moltiplicazioni né tabelle. Richiede fine-tuning.
2. Pesi su reticoli non cubici (esagonale / Eisenstein): più precisione per bit.
3. Caricare un modello BitNet reale e misurare token/s end-to-end.
