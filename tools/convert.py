"""Converte microsoft/bitnet-b1.58-2B-4T (safetensors, pesi impacchettati 2 bit)
nel formato .aim: pesi ternari in base 3 (5 per byte) nel layout a tile usato
dal kernel AVX2, embedding/lm_head (tied) in int8 per riga, norm in float32.

Uso: python tools/convert.py [out.aim]
"""
import json, struct, sys, os, time
import numpy as np
from huggingface_hub import hf_hub_download

REPO = "microsoft/bitnet-b1.58-2B-4T"
TILE, GROUP = 32, 5
POW3 = np.array([1, 3, 9, 27, 81], dtype=np.uint8)


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


def write_t3(f, t, row_scales):
    rows, cols = t.shape
    data, rows_pad = pack_t3_tiled(t)
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
    out_path = sys.argv[1] if len(sys.argv) > 1 else "models/bitnet-2b-4t.aim"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cfg = json.load(open(hf_hub_download(REPO, "config.json")))
    st = load_safetensors(hf_hub_download(REPO, "model.safetensors"))

    L, H, I = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["intermediate_size"]
    NH, NKV, V = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["vocab_size"]
    HD = H // NH
    t0 = time.time()
    total = 0
    with open(out_path, "wb") as f:
        f.write(b"AIMODEL1")
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

        for l in range(L):
            p = "model.layers.%d." % l
            total += write_f32(f, as_f32(st[p + "input_layernorm.weight"]))
            q, qs = tern(p + "self_attn.q_proj"); k, ks = tern(p + "self_attn.k_proj"); v, vs = tern(p + "self_attn.v_proj")
            assert q.shape == (NH * HD, H) and k.shape == (NKV * HD, H)
            total += write_t3(f, np.concatenate([q, k, v]), np.concatenate([qs, ks, vs]))
            total += write_f32(f, as_f32(st[p + "self_attn.attn_sub_norm.weight"]))
            o, os_ = tern(p + "self_attn.o_proj"); total += write_t3(f, o, os_)
            total += write_f32(f, as_f32(st[p + "post_attention_layernorm.weight"]))
            g, gs = tern(p + "mlp.gate_proj"); u, us = tern(p + "mlp.up_proj")
            total += write_t3(f, np.concatenate([g, u]), np.concatenate([gs, us]))
            total += write_f32(f, as_f32(st[p + "mlp.ffn_sub_norm.weight"]))
            d, ds = tern(p + "mlp.down_proj"); total += write_t3(f, d, ds)
            print("layer %2d/%d  (%.0f s)" % (l + 1, L, time.time() - t0), flush=True)
    print("scritto %s: %.1f MB" % (out_path, total / 1e6))


if __name__ == "__main__":
    main()
