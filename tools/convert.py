"""Converte microsoft/bitnet-b1.58-2B-4T (safetensors, pesi impacchettati 2 bit)
nel formato .aim: pesi ternari in base 3 (5 per byte) nel layout a tile usato
dal kernel AVX2, embedding/lm_head (tied) in int8 per riga, norm in float32.

Uso: python tools/convert.py [out.aim]
"""
import json, struct, sys, os, time
import numpy as np
from huggingface_hub import hf_hub_download

REPO = "microsoft/bitnet-b1.58-2B-4T"
REPO_BF16 = "microsoft/bitnet-b1.58-2B-4T-bf16"
TILE, GROUP = 32, 5
POW3 = np.array([1, 3, 9, 27, 81], dtype=np.uint8)


class Code:
    """Codice a byte generico, stessa enumerazione di src/codes.c:
    n pesi ternari, al piu' kmax non nulli; indice base-3 con la cifra 0 = peso 0."""
    def __init__(self, name):
        self.name = name
        if name == "t5": self.n, self.kmax = 5, 5
        elif name.startswith("s") and len(name) == 3: self.n, self.kmax = int(name[1]), int(name[2])
        else: raise ValueError(name)
        total = 3 ** self.n
        idx = np.arange(total)
        digits = np.stack([(idx // 3 ** i) % 3 - 1 for i in range(self.n)], axis=1).astype(np.int8)   # [total, n]
        nz = (digits != 0).sum(axis=1)
        ok = nz <= self.kmax
        self.enc = np.full(total, -1, dtype=np.int16)
        self.enc[ok] = np.arange(ok.sum())
        self.ncodes = int(ok.sum())
        assert self.ncodes <= 256, (name, self.ncodes)
        self.pow3 = (3 ** np.arange(self.n)).astype(np.int32)

    def pack(self, t, mag):
        """t: (rows, cols) int8 in {-1,0,1}; mag: (rows, cols) float |w| per la potatura.
        Ritorna (bytes del layout a tile, rows_pad, n_potati)."""
        rows, cols = t.shape
        cols_pad = (cols + self.n - 1) // self.n * self.n
        rows_pad = (rows + TILE - 1) // TILE * TILE
        G = cols_pad // self.n
        tp = np.zeros((rows_pad, cols_pad), dtype=np.int8); tp[:rows, :cols] = t
        mp = np.zeros((rows_pad, cols_pad), dtype=np.float32); mp[:rows, :cols] = mag
        tg = tp.reshape(rows_pad, G, self.n); mg = mp.reshape(rows_pad, G, self.n)
        pruned = 0
        if self.kmax < self.n:
            nz = (tg != 0)
            key = np.where(nz, mg, -1.0)                       # i non nulli ordinati per |w|
            rank = np.argsort(np.argsort(-key, axis=2), axis=2)  # 0 = |w| piu' grande
            drop = nz & (rank >= self.kmax)
            pruned = int(drop.sum())
            tg = np.where(drop, 0, tg)
        idx = ((tg.astype(np.int32) + 1) * self.pow3).sum(axis=2)
        codes = self.enc[idx]
        assert (codes >= 0).all()
        tiled = codes.astype(np.uint8).reshape(rows_pad // TILE, TILE, G).transpose(0, 2, 1)
        return np.ascontiguousarray(tiled).tobytes(), rows_pad, pruned


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    tensors = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        tensors[k] = (v["dtype"], v["shape"], mm[base + a: base + b])
    return tensors


def as_f32(t):
    dtype, shape, raw = t
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return u16.view(np.float32).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    raise ValueError(dtype)


def as_u8(t):
    dtype, shape, raw = t
    assert dtype == "U8"
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape)


def unpack_ternary(packed):
    """(out/4, in) uint8 -> (out, in) int8 in {-1,0,1}, come transformers.integrations.bitnet"""
    pr = packed.shape[0]
    out = np.empty((pr * 4, packed.shape[1]), dtype=np.int8)
    for i in range(4):
        out[i * pr:(i + 1) * pr] = ((packed >> (2 * i)) & 3).astype(np.int8) - 1
    return out


def pack_t3_tiled(t):
    """(rows, cols) int8 in {-1,0,1} -> bytes del layout a tile: [tile][gruppo][32 righe]"""
    rows, cols = t.shape
    cols_pad = (cols + GROUP - 1) // GROUP * GROUP
    rows_pad = (rows + TILE - 1) // TILE * TILE
    G = cols_pad // GROUP
    tp = np.zeros((rows_pad, cols_pad), dtype=np.uint8)
    tp[:rows, :cols] = (t + 1).astype(np.uint8)
    codes = (tp.reshape(rows_pad, G, GROUP) * POW3).sum(axis=2, dtype=np.uint16).astype(np.uint8)
    tiled = codes.reshape(rows_pad // TILE, TILE, G).transpose(0, 2, 1)
    return np.ascontiguousarray(tiled).tobytes(), rows_pad


def write_t3(f, t, row_scales, code=None, mag=None, stats=None):
    rows, cols = t.shape
    if code is None:
        data, rows_pad = pack_t3_tiled(t)
    else:
        data, rows_pad, pruned = code.pack(t, mag)
        if stats is not None:
            stats["pruned"] += pruned; stats["nonzero"] += int((t != 0).sum()); stats["total"] += t.size
    sc = np.zeros(rows_pad, dtype=np.float32)
    sc[:rows] = row_scales
    f.write(struct.pack("<ii", rows, cols))
    f.write(data)
    f.write(sc.tobytes())
    return len(data) + rows_pad * 4


def write_f32(f, a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    f.write(a.tobytes())
    return a.nbytes


def main():
    args = sys.argv[1:]
    code = None
    if "--code" in args:
        i = args.index("--code"); code = Code(args[i + 1]); del args[i:i + 2]
    out_path = args[0] if args else "models/bitnet-2b-4t.aim"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cfg = json.load(open(hf_hub_download(REPO, "config.json")))
    st = load_safetensors(hf_hub_download(REPO, "model.safetensors"))
    st_bf = load_safetensors(hf_hub_download(REPO_BF16, "model.safetensors")) if code and code.kmax < code.n else None
    stats = {"pruned": 0, "nonzero": 0, "total": 0}

    L, H, I = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["intermediate_size"]
    NH, NKV, V = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["vocab_size"]
    HD = H // NH
    t0 = time.time()
    total = 0
    with open(out_path, "wb") as f:
        if code is None:
            f.write(b"AIMODEL1")
        else:
            f.write(b"AIMODEL2"); f.write(code.name.ljust(8).encode())
        f.write(struct.pack("<8i", L, H, I, NH, NKV, HD, V, cfg["max_position_embeddings"]))
        f.write(struct.pack("<2f", cfg["rms_norm_eps"], cfg["rope_theta"]))

        # embedding (tied lm_head): int8 per riga, absmax
        emb = st["model.embed_tokens.weight"]
        q_all = np.empty((V, H), dtype=np.int8)
        s_all = np.empty(V, dtype=np.float32)
        CH = 8192
        for r0 in range(0, V, CH):
            blk = as_f32((emb[0], [min(CH, V - r0), H], emb[2][r0 * H * 2:(r0 + min(CH, V - r0)) * H * 2]))
            amax = np.abs(blk).max(axis=1)
            s = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
            q_all[r0:r0 + blk.shape[0]] = np.clip(np.rint(blk / s[:, None]), -127, 127).astype(np.int8)
            s_all[r0:r0 + blk.shape[0]] = s
        f.write(q_all.tobytes()); f.write(s_all.tobytes())
        total += q_all.nbytes + s_all.nbytes
        print("embedding int8: %.1f MB" % (q_all.nbytes / 1e6), flush=True)

        total += write_f32(f, as_f32(st["model.norm.weight"]))

        def tern(name):
            w = unpack_ternary(as_u8(st[name + ".weight"]))
            ws = float(as_f32(st[name + ".weight_scale"])[0])
            # AutoBitLinear (offline): output = linear(x_q, ternary) * weight_scale
            return w, np.full(w.shape[0], ws, dtype=np.float32)

        def mag(name):
            """|w| dai pesi master bf16, per decidere cosa potare"""
            if st_bf is None: return None
            return np.abs(as_f32(st_bf[name + ".weight"]))

        def wt3(t, sc, names):
            m = None
            if st_bf is not None:
                m = np.concatenate([mag(n) for n in names])
            return write_t3(f, t, sc, code, m, stats)

        if st_bf is not None:   # sanita': i ternari del checkpoint coincidono con round(w/mean|w|) dei master?
            w0 = as_f32(st_bf["model.layers.0.self_attn.q_proj.weight"]); t0_ = unpack_ternary(as_u8(st["model.layers.0.self_attn.q_proj.weight"]))
            s0 = np.abs(w0).mean(); tq = np.clip(np.rint(w0 / s0), -1, 1).astype(np.int8)
            print("layer0 q_proj: mean|w_bf16| = %.4f, weight_scale = %.4f, accordo ternario bf16 vs packed = %.4f" %
                  (s0, float(as_f32(st["model.layers.0.self_attn.q_proj.weight_scale"])[0]), (tq == t0_).mean()), flush=True)

        for l in range(L):
            p = "model.layers.%d." % l
            total += write_f32(f, as_f32(st[p + "input_layernorm.weight"]))
            q, qs = tern(p + "self_attn.q_proj"); k, ks = tern(p + "self_attn.k_proj"); v, vs = tern(p + "self_attn.v_proj")
            assert q.shape == (NH * HD, H) and k.shape == (NKV * HD, H)
            total += wt3(np.concatenate([q, k, v]), np.concatenate([qs, ks, vs]), [p + "self_attn.q_proj", p + "self_attn.k_proj", p + "self_attn.v_proj"])
            total += write_f32(f, as_f32(st[p + "self_attn.attn_sub_norm.weight"]))
            o, os_ = tern(p + "self_attn.o_proj"); total += wt3(o, os_, [p + "self_attn.o_proj"])
            total += write_f32(f, as_f32(st[p + "post_attention_layernorm.weight"]))
            g, gs = tern(p + "mlp.gate_proj"); u, us = tern(p + "mlp.up_proj")
            total += wt3(np.concatenate([g, u]), np.concatenate([gs, us]), [p + "mlp.gate_proj", p + "mlp.up_proj"])
            total += write_f32(f, as_f32(st[p + "mlp.ffn_sub_norm.weight"]))
            d, ds = tern(p + "mlp.down_proj"); total += wt3(d, ds, [p + "mlp.down_proj"])
            print("layer %2d/%d  (%.0f s)" % (l + 1, L, time.time() - t0), flush=True)
    print("scritto %s: %.1f MB" % (out_path, total / 1e6))
    if stats["total"]:
        print("codice %s: potati %d pesi su %d non nulli (%.2f%%), densita' %.3f -> %.3f" % (
            code.name, stats["pruned"], stats["nonzero"], 100.0 * stats["pruned"] / max(1, stats["nonzero"]),
            stats["nonzero"] / stats["total"], (stats["nonzero"] - stats["pruned"]) / stats["total"]))


if __name__ == "__main__":
    main()
