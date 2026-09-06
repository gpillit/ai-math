# ai-math

Laboratorio in C per **nuove strutture algebriche orientate all'inferenza LLM
memory-bound** (tutto il modello in RAM, calcolo su CPU).

Stato: un modello reale (BitNet b1.58 2B-4T) gira end-to-end sopra la
struttura ternaria in base 3, in 750 MB di RAM, a **30-34 token/s** in
generazione e **90-105 token/s** in prefill su un portatile (Core Ultra 7
155U, 15 W). Risposte corrette e coerenti.

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

## Struttura 1b: la stessa matrice, due algebre di calcolo

Il layout in RAM è uno solo (base 3, tile di 32 righe). Sopra ci girano due
kernel diversi a seconda del regime:

- **decode (1 token)**: memory-bound → LUT via `vpshufb`, mai una
  moltiplicazione (`src/ternary_fast.c`);
- **prefill (B token)**: compute-bound → ogni tile viene spacchettato una
  volta in int8 (colonne interleaved a 4) e moltiplicato per tutti i B token
  con `vpdpbusd` (AVX-VNNI), correzione `128·Σw` con lo stesso kernel
  (`src/ternary_gemm.c`). Nel bench: 360-500 GOPS, il limite del chip.

## Cosa c'è

| file | contenuto |
|---|---|
| `include/aim.h` | API: semianello generico, matrice ternaria, layout a tile, thread pool |
| `src/semiring.c` | GEMV parametrico su (⊕,⊗): reale, tropicale (max,+), log (lse,+) |
| `src/ternary.c` | quantizzazione TWN, packing base 3, kernel LUT di riferimento |
| `src/ternary_fast.c` | kernel GEMV: blocking di cache, fattorizzazione 27×9, AVX2 a tile |
| `src/ternary_gemm.c` | kernel GEMM per il prefill: unpack a int8 per tile + AVX-VNNI |
| `src/pool.c` | thread pool con barriere a spin e affinità ai core (libgomp costava 150 µs a chiamata) |
| `src/int8.c` | GEMV int8 per l'lm_head |
| `src/model.c`, `include/aim_model.h` | forward BitNet: RMSNorm, RoPE, GQA con KV cache, relu², sub-norm |
| `app/run.c` | generazione greedy da id di token, statistiche |
| `tools/convert.py` | HuggingFace safetensors → formato `.aim` (ternario a tile, embedding int8) |
| `tools/chat.py` | tokenizza con chat template, lancia `run`, decodifica |
| `tools/ref_forward.py` | forward di riferimento in numpy per verificare il C layer per layer |
| `tools/ppl.py` | perplexity su quattro testi, con il semianello dell'attention scelto |
| `tools/longctx.py` | decode a contesto lungo: ms/token dell'attention per semianello |
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
ibride; su questo portatile 10 è il migliore), `AIM_BATCH` (token per batch
nel prefill, default 32), `AIM_DUMP=1` stampa stato e top-5 per posizione
(confrontabile con `tools/ref_forward.py`; forza batch 1).

## Risultati

### Modello intero: BitNet b1.58 2B-4T, 30 layer, 2.4 miliardi di parametri

| | |
|---|---|
| RAM | 750 MB (417 MB ternari a 1.60 bit/peso + 328 MB embedding/lm_head int8) |
| decode | **30.7 ms/token (32.6 tok/s)** con 12 thread, 29.4 ms con 10 |
| di cui | GEMV ternari 20-22 ms (19-21 GB/s), lm_head 8.3 ms (39 GB/s, al tetto), attention 0.5 ms |
| prefill | **9.6-11 ms/token (90-105 tok/s)** con batch 64 (`AIM_BATCH`, default 32); batch 1: 32 tok/s |
| di cui | GEMM 8-9.5 ms, attention 0.7-1 ms (AVX2, cresce col contesto) |

### Kernel isolato, 4096×4096 (un GEMV = un token su un layer)

| kernel | byte pesi | ms | GB/s eff. | vs fp32 |
|---|---|---|---|---|
| fp32 | 67 MB | 1.56 | 43 | 1× |
| ternario, LUT T[243] scalare | 3.4 MB | 0.49 | 7 | 3.2× |
| ternario, LUT con blocking L1 | 3.4 MB | 0.40 | 8.4 | 3.9× |
| ternario, 27×9 AVX2 a tile | 3.4 MB | **0.185** | 18 | **8.4×** |
| GEMM VNNI, 32 token | 3.4 MB | 2.1 (0.066/token) | | 508 GOPS |

Regime DRAM (54 MB): 1.42 ms, **38 GB/s**, cioè il tetto di banda della
macchina. Il kernel isolato è RAM-bound; nel modello le matrici sono più
piccole (1.3-7 MB) e la coda degli E-core lenti costa ancora un fattore ~2.

## Esperimento 1: il semianello dell'attention

L'attention è un prodotto in un semianello: i punteggi q·k si combinano con
⊕ e i valori si pesano di conseguenza. Con (logsumexp, +) è la softmax;
con (max, +) è l'attention tropicale, che prende solo l'argmax. Tra i due
c'è la famiglia continua di Maslov, ⊕ₕ(a,b) = h·log(e^{a/h} + e^{b/h}):
h = 1 è la softmax, h → 0 è il max. Tutto questo si cambia a inferenza,
senza riaddestrare (`AIM_ATTN=softmax|max|topk=K|h=X`, `tools/ppl.py`).

Perplexity di BitNet 2B-4T su quattro testi di generi diversi
(`tools/ppl.py --all`):

| testo | token | softmax | (max,+) | top-8 | top-32 | h=0.85 | h=0.7 |
|---|---|---|---|---|---|---|---|
| storia | 299 | 6.43 | 20.55 | 6.55 | **6.40** | 6.39 | 6.61 |
| codice | 310 | 1.76 | 3.98 | 1.81 | **1.76** | 1.78 | 1.80 |
| dialogo | 227 | 11.09 | 36.96 | 11.83 | **10.99** | 11.32 | 11.69 |
| scienza | 246 | 4.99 | 16.05 | 5.33 | **4.98** | 5.06 | 5.23 |

Scansione della famiglia di Maslov sul testo "storia":

| h | 0.25 | 0.5 | 0.7 | 0.85 | 1.0 | 1.15 | 1.3 | 1.6 | 2.0 |
|---|---|---|---|---|---|---|---|---|---|
| ppl | 8.26 | 7.00 | 6.61 | 6.39 | 6.43 | 6.54 | 6.68 | 7.28 | 11.55 |

Cosa dice:

1. **Il semianello tropicale puro non è sostituibile a inferenza** in un
   modello addestrato con la softmax: perplexity da 2.3× a 3.3× su tutti i
   testi.
2. **La massa dell'attention è concentrata, ma non in 8 chiavi**: top-8
   costa dall'1.7% (storia) al 7% (dialogo, scienza). **Top-32 è uguale o
   leggermente migliore della softmax su tutti e quattro i testi** (da
   −0.1% a −0.9%): tagliare la coda lunga dei punteggi non toglie niente e
   forse toglie rumore.
3. **h = 0.85 non generalizza**: batte la softmax solo su "storia" e perde
   sugli altri tre. Era un effetto del singolo testo. Il modello sta dove è
   stato addestrato, h = 1, e la curva è asimmetrica: verso il max (h → 0)
   degrada lentamente, verso l'uniforme (h = 2) crolla.

Il top-k a inferenza non è nuovo (attention sparsa, Quest e simili); la
scansione della famiglia di Maslov come misura di "quanto tropicale" può
essere un modello è la parte da approfondire.

### Top-k applicato alla KV cache, a contesto lungo

Con 2048 token di contesto la KV cache (315 MB in fp32) supera i pesi
ternari (417 MB) come termine di banda per token. Decode a 2048 token,
misurato con `tools/longctx.py`:

| attention | attention ms/token | decode tok/s | prefill tok/s |
|---|---|---|---|
| softmax, kernel iniziale (scalare, un item per head) | 45.1 | 10.2 | 38.8 |
| softmax, kernel a 3 fasi (GQA 4 head per riga K, exp AVX2, chunk di 256 posizioni, cache [layer][kv-head][pos]) | 25.8 | 14.1 | 51.9 |
| top-32, kernel a 3 fasi | **10.1** | **18.8** | **63.0** |

Con top-32 la fase 3 legge solo le 32 righe di V che contano; K va letta
tutta per calcolare i punteggi (10 ms ≈ 5 MB/layer di K a ~15 GB/s). Per
andare oltre servono K in fp16 o un indice sulle chiavi (stile Quest).
Il kernel è verificato contro il forward numpy: stessi top-5 in ogni
posizione, logit entro l'arrotondamento int8.

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
**Il packing di 5 trit per byte esiste già**: è il tipo `TQ1_0` di ggml /
llama.cpp (1.69 bit/peso), che decodifica moltiplicando per 3 e leggendo il
riporto. Quello che lì non c'è è la fattorizzazione Z₃⁵ = Z₃³ × Z₃² della
tabella di lookup (27×9) con `vpshufb`: è la parte da verificare in
letteratura prima di chiamarla nuova.

## Prossimi passi candidati

1. Confronto diretto con bitnet.cpp sulla stessa macchina. Richiede cmake e
   clang (non installati) e il suo generatore di kernel; non ancora fatto.
2. KV cache in fp16 (F16C c'è) e indice sulle chiavi per non leggere tutta
   K con il top-k; esperimento 1 su un modello più grande.
3. Cambio di semianello nell'MLP (relu² in (max,+)): richiede fine-tuning,
   quindi PyTorch e un modello piccolo.
4. Pesi su reticoli non cubici (esagonale / Eisenstein): più precisione per bit.
