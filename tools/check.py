"""Riferimento numpy per `build/run model.aim --check`: layer 0 qkv su x deterministica."""
import sys, numpy as np
sys.path.insert(0, "tools")
from convert import load_safetensors, as_f32, as_u8, unpack_ternary
from huggingface_hub import hf_hub_download
st = load_safetensors(hf_hub_download("microsoft/bitnet-b1.58-2B-4T", "model.safetensors"))
p = "model.layers.0.self_attn."
ws = {n: float(as_f32(st[p + n + ".weight_scale"])[0]) for n in ("q_proj", "k_proj", "v_proj")}
print("weight_scale:", ws)
W = np.concatenate([unpack_ternary(as_u8(st[p + n + ".weight"])) for n in ("q_proj", "k_proj", "v_proj")]).astype(np.float32)
vals, cnt = np.unique(W, return_counts=True); print("trit:", dict(zip(vals.tolist(), (cnt / cnt.sum()).round(3).tolist())))
scale = np.concatenate([np.full(2560, 1 / ws["q_proj"]), np.full(640, 1 / ws["k_proj"]), np.full(640, 1 / ws["v_proj"])]).astype(np.float32)
x = ((np.arange(2560) * 37) % 255 - 127).astype(np.float32)
y = (W @ x) * scale
print("y[0..5] =", y[:6].round(4), " sum=%.4f" % y.sum())
emb = st["model.embed_tokens.weight"]; row = as_f32((emb[0], [1, 2560], emb[2][:2560 * 2]))[0]
print("emb[0][0..3] float:", row[:4], "absmax", np.abs(row).max())
tok_ids = __import__("transformers").AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T")("User: hi")["input_ids"]
print("ids('User: hi') =", tok_ids)
