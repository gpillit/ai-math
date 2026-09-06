"""Decode a contesto lungo: ms/token dell'attention con softmax vs top-k.

Uso: python tools/longctx.py [--ctx 2048] [--attn softmax] [-n 32]
"""
import subprocess, sys, os
from transformers import AutoTokenizer
sys.path.insert(0, "tools")
from ppl import TEXTS, REPO


def main():
    args = sys.argv[1:]
    ctx = 2048; attn = "softmax"; n = 32
    if "--ctx" in args: i = args.index("--ctx"); ctx = int(args[i + 1])
    if "--attn" in args: i = args.index("--attn"); attn = args[i + 1]
    if "-n" in args: i = args.index("-n"); n = int(args[i + 1])
    tok = AutoTokenizer.from_pretrained(REPO)
    body = " ".join(TEXTS.values())
    ids = tok(body)["input_ids"]
    while len(ids) < ctx - n:
        ids = ids + tok(body)["input_ids"][1:]
    ids = ids[:ctx - n]
    exe = os.path.join("build", "run.exe" if os.name == "nt" else "run")
    env = dict(os.environ, AIM_ATTN=attn, AIM_BATCH="64")
    r = subprocess.run([exe, "models/bitnet-2b-4t.aim", "--ids", ",".join(map(str, ids)), "-n", str(n), "--ctx", str(ctx)],
                       capture_output=True, text=True, env=env)
    for line in r.stderr.splitlines():
        if line.startswith(("prefill", "decode")):
            print(line)


if __name__ == "__main__":
    main()
