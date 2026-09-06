"""Tokenizza un prompt (chat template BitNet/Llama-3), lancia build/run e decodifica.

Uso: python tools/chat.py "domanda" [-n 64] [--raw]   (--raw: senza chat template)
"""
import subprocess, sys, os, time
from transformers import AutoTokenizer

REPO = "microsoft/bitnet-b1.58-2B-4T"


def main():
    args = sys.argv[1:]
    n_new, raw = 64, False
    if "-n" in args:
        i = args.index("-n"); n_new = int(args[i + 1]); del args[i:i + 2]
    if "--raw" in args:
        args.remove("--raw"); raw = True
    prompt = " ".join(args) or "What is the capital of France?"

    tok = AutoTokenizer.from_pretrained(REPO)
    if raw:
        ids = tok(prompt)["input_ids"]
    else:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        ids = tok(text)["input_ids"]
    eos = [tok.eos_token_id] + [tok.convert_tokens_to_ids(t) for t in ("<|eot_id|>",) if t in tok.get_vocab()]

    exe = os.path.join("build", "run.exe" if os.name == "nt" else "run")
    cmd = [exe, "models/bitnet-2b-4t.aim", "--ids", ",".join(map(str, ids)), "-n", str(n_new),
           "--eos", ",".join(map(str, eos))]
    print("prompt ids:", len(ids), file=sys.stderr)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    out = []
    for line in p.stdout:
        t = int(line.strip()); out.append(t)
        sys.stdout.write(tok.decode([t])); sys.stdout.flush()
    p.wait()
    print()
    print("---", file=sys.stderr)
    print("decodificato:", repr(tok.decode(out)), file=sys.stderr)


if __name__ == "__main__":
    main()
