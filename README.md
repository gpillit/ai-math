# ai-math

Laboratorio in C per **nuove strutture algebriche orientate all'inferenza LLM
memory-bound** (tutto il modello in RAM, calcolo su CPU).

Stato: un modello reale (BitNet b1.58 2B-4T) gira end-to-end sopra la
struttura ternaria in base 3, in 750 MB di RAM, a **30-34 token/s** su un
portatile (Core Ultra 7 155U, 15 W). Risposte corrette e coerenti.

## Il vincolo fisico che guida tutto

Nell'inferenza autoregressiva ogni token deve leggere *tutti* i pesi dalla RAM.
Su una CPU il calcolo è quasi gratis; la banda verso la memoria no.

    tempo per token  ≈  (byte dei pesi) / (GB/s della RAM)

Quindi una struttura algebrica "buona" per i pesi deve:

1. minimizzare i **bit per peso** effettivamente letti;
2. trasformare `peso ⊗ attivazione` in una **lookup**, non in una moltiplicazione;
3. restare un modulo su un semianello, così attention e MLP restano esprimibili.

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
| `include/aim.h` | API: semianello generico, matrice ternaria, layout a tile, thread pool |
| `src/semiring.c` | GEMV parametrico su (⊕,⊗): reale, tropicale (max,+), log (lse,+) |
| `src/ternary.c` | quantizzazione TWN, packing base 3, kernel LUT di riferimento |
| `src/ternary_fast.c` | kernel: blocking di cache, fattorizzazione 27×9, AVX2 a tile |
| `src/pool.c` | thread pool con barriere a spin e affinità ai core (libgomp costava 150 µs a chiamata) |
| `src/int8.c` | GEMV int8 per l'lm_head |
| `src/model.c`, `include/aim_model.h` | forward BitNet: RMSNorm, RoPE, GQA con KV cache, relu², sub-norm |
| `app/run.c` | generazione greedy da id di token, statistiche |
| `tools/convert.py` | HuggingFace safetensors → formato `.aim` (ternario a tile, embedding int8) |
| `tools/chat.py` | tokenizza con chat template, lancia `run`, decodifica |
| `tools/ref_forward.py` | forward di riferimento in numpy per verificare il C layer per layer |
| `tests/test.c` | 16 test: biiezione della codifica, kernel == fp32, semianelli, esattezza SIMD |
| `bench/bench.c` | banda effettiva e compressione dei kernel |

## Build e uso

Windows (MSYS2 ucrt64): `./build.sh` poi `./build/test`, `./build/bench`.
Linux: `make`.

Modello (una volta, scarica 1.2 GB da HuggingFace e scrive 750 MB):

    python -m venv .venv && .venv/Scripts/pip install numpy safetensors huggingface_hub transformers jinja2
    .venv/Scripts/python tools/convert.py models/bitnet-2b-4t.aim

Chat:

    .venv/Scripts/python tools/chat.py "Explain why the sky is blue." -n 80

Variabili: `AIM_THREADS` (default: tutti i core meno i 2 LP-E sulle CPU
ibride; su questo portatile 10 è il migliore), `AIM_DUMP=1` stampa stato e
top-5 per posizione (confrontabile con `tools/ref_forward.py`).

## Risultati

### Modello intero: BitNet b1.58 2B-4T, 30 layer, 2.4 miliardi di parametri

| | |
|---|---|
| RAM | 750 MB (417 MB ternari a 1.60 bit/peso + 328 MB embedding/lm_head int8) |
| decode | **30.7 ms/token (32.6 tok/s)** con 12 thread, 29.4 ms con 10 |
| di cui | GEMV ternari 20-22 ms (19-21 GB/s), lm_head 8.3 ms (39 GB/s, al tetto), attention 0.5 ms |
| prefill | un token alla volta, stessa velocità del decode |

### Kernel isolato, 4096×4096 (un GEMV = un token su un layer)

| kernel | byte pesi | ms | GB/s eff. | vs fp32 |
|---|---|---|---|---|
| fp32 | 67 MB | 1.56 | 43 | 1× |
| ternario, LUT T[243] scalare | 3.4 MB | 0.49 | 7 | 3.2× |
| ternario, LUT con blocking L1 | 3.4 MB | 0.40 | 8.4 | 3.9× |
| ternario, 27×9 AVX2 a tile | 3.4 MB | **0.185** | 18 | **8.4×** |

Regime DRAM (54 MB): 1.42 ms, **38 GB/s**, cioè il tetto di banda della
macchina. Il kernel isolato è RAM-bound; nel modello le matrici sono più
piccole (1.3-7 MB) e la coda degli E-core lenti costa ancora un fattore ~2.

## Cose imparate (che non si trovano nei paper)

- **libgomp su Windows/mingw costa 50 µs per regione parallela vuota e 150 µs
  con barriere.** Con 120 GEMV per token erano 18 ms su 44. Un pool
  persistente con barriere a spin costa 2-3 µs.
- **Le CPU ibride vanno pilotate a mano.** Windows riporta E-core e LP-E-core
  nella stessa classe di efficienza; i 2 LP-E vengono parcheggiati e un
  thread che ci finisce sopra blocca tutti alla barriera (14 thread: 4 s per
  token; 12 thread con affinità fissa: 30 tok/s).
- **Su un chip da 15 W i thread che spinnano rubano potenza a quelli che
  lavorano**: spin breve poi `WaitOnAddress`.
- **Semantica di `weight_scale` in BitNet**: la classe `AutoBitLinear`
  (offline) *moltiplica* per `weight_scale`; la classe `BitLinear` *divide*.
  Con il segno sbagliato il modello produce " the, the, the" invece di
  risposte: l'unico modo per accorgersene è stato un forward di riferimento
  in numpy confrontato layer per layer.

## Metodo di lavoro

Per ogni idea: (1) assiomi della struttura, (2) kernel in C, (3) misura di
bit/peso, GB/s, errore, (4) verifica in letteratura che non esista già.
Si chiama "nuova" solo dopo il punto 4 e solo se batte le baseline.

Vicini di casa da conoscere (non nuovi): LUT-GEMM, T-MAC, BitNet b1.58 e i
kernel TL1/TL2 di bitnet.cpp, quantizzazione su reticoli (QuIP#, AQLM).
La fattorizzazione 27×9 della tabella base-3 è la parte da verificare in
letteratura prima di chiamarla nuova.

## Prossimi passi candidati

1. Prefill a batch (GEMM invece di GEMV): i pesi si leggono una volta per
   tutto il prompt invece che per ogni token.
2. Confronto diretto con bitnet.cpp sulla stessa macchina.
3. Cambio di semianello dentro il modello: layer in (max,+), niente
   moltiplicazioni né tabelle. Richiede fine-tuning.
4. Pesi su reticoli non cubici (esagonale / Eisenstein): più precisione per bit.
