"""Riferimento numpy del forward BitNet per UN solo token (posizione 0).
Stampa per ogni layer x[0..3] e sum|x|, come `AIM_DUMP=1 build/run`.
Uso: python tools/ref_forward.py [token_id]
"""
import sys, json, time
import numpy as np
sys.path.insert(0, "tools")
from convert import load_safetensors, as_f32, as_u8, unpack_ternary
from huggingface_hub import hf_hub_download

REPO = "microsoft/bitnet-b1.58-2B-4T"
cfg = json.load(open(hf_hub_download(REPO, "config.json")))
st = load_safetensors(hf_hub_download(REPO, "model.safetensors"))
H, I, NH, NKV = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
HD, eps = H // NH, cfg["rms_norm_eps"]


def rms(x, w):
    return x / np.sqrt((x * x).mean() + eps) * w


def bitlinear(name, x):
    W = unpack_ternary(as_u8(st[name + ".weight"])).astype(np.float32)
    ws = float(as_f32(st[name + ".weight_scale"])[0])
    s = 127.0 / max(np.abs(x).max(), 1e-5)
    xq = np.clip(np.rint(x * s), -128, 127)
    return (W @ xq) / s * ws     # AutoBitLinear offline: * weight_scale



def rope(v, pos, nh):
    half = HD // 2
    inv = cfg["rope_theta"] ** (-2.0 * np.arange(half) / HD)
    cs, sn = np.cos(pos * inv).astype(np.float32), np.sin(pos * inv).astype(np.float32)
    v = v.reshape(nh, HD).copy()
    a, b = v[:, :half].copy(), v[:, half:].copy()
    v[:, :half] = a * cs - b * sn
    v[:, half:] = b * cs + a * sn
    return v


if len(sys.argv) > 1 and not sys.argv[1].isdigit():
    from transformers import AutoTokenizer
    ids = AutoTokenizer.from_pretrained(REPO)(" ".join(sys.argv[1:]))["input_ids"]
else:
    ids = [int(a) for a in sys.argv[1:]] or [128000]
print("ids:", ",".join(map(str, ids)))
emb = st["model.embed_tokens.weight"]
E = as_f32((emb[0], [cfg["vocab_size"], H], emb[2]))
L = cfg["num_hidden_layers"]
Kc = [[] for _ in range(L)]; Vc = [[] for _ in range(L)]
group = NH // NKV
t0 = time.time()
for pos, tok in enumerate(ids):
    x = E[tok].copy()
    for l in range(L):
        p = "model.layers.%d." % l
        xb = rms(x, as_f32(st[p + "input_layernorm.weight"]))
        q = rope(bitlinear(p + "self_attn.q_proj", xb), pos, NH)
        k = rope(bitlinear(p + "self_attn.k_proj", xb), pos, NKV)
        v = bitlinear(p + "self_attn.v_proj", xb).reshape(NKV, HD)
        Kc[l].append(k); Vc[l].append(v)
        K = np.stack(Kc[l]); Vv = np.stack(Vc[l])          # [T, NKV, HD]
        attn = np.empty((NH, HD), dtype=np.float32)
        for h in range(NH):
            kh = h // group
            sc = K[:, kh] @ q[h] / np.sqrt(HD)
            sc = np.exp(sc - sc.max()); sc /= sc.sum()
            attn[h] = sc @ Vv[:, kh]
        attn = rms(attn.reshape(-1), as_f32(st[p + "self_attn.attn_sub_norm.weight"]))
        x = x + bitlinear(p + "self_attn.o_proj", attn)
        xb = rms(x, as_f32(st[p + "post_attention_layernorm.weight"]))
        g = bitlinear(p + "mlp.gate_proj", xb); u = bitlinear(p + "mlp.up_proj", xb)
        h = rms(np.maximum(g, 0) ** 2 * u, as_f32(st[p + "mlp.ffn_sub_norm.weight"]))
        x = x + bitlinear(p + "mlp.down_proj", h)
        if pos == 0 or l in (0, L - 1):
            print("pos %d layer %2d: x[0..3] = %9.4f %9.4f %9.4f %9.4f  sum|x| = %.3f" % (pos, l, x[0], x[1], x[2], x[3], np.abs(x).sum()), flush=True)
    logits = E @ rms(x, as_f32(st["model.norm.weight"]))
    top = np.argsort(-logits)[:5]
    print("pos %d top5: %s %s (%.0f s)" % (pos, top.tolist(), logits[top].round(3).tolist(), time.time() - t0), flush=True)
