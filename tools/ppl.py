"""Perplexity del modello su un testo, con il semianello dell'attention scelto.

Uso: python tools/ppl.py [--attn softmax|max|topk=K] [--text file.txt]
"""
import subprocess, sys, os
from transformers import AutoTokenizer

REPO = "microsoft/bitnet-b1.58-2B-4T"
DEFAULT_TEXT = """The Industrial Revolution began in Britain in the late eighteenth century and spread across Europe and North America during the nineteenth. It transformed economies that had been based on agriculture and handicrafts into economies based on large-scale industry, mechanized manufacturing, and the factory system. New machines, new power sources, and new ways of organizing work made existing industries more productive and efficient. The steam engine, first developed to pump water out of mines, was soon adapted to drive machinery in textile mills and, later, locomotives and ships. Iron and steel production grew rapidly as coal replaced charcoal as the main fuel for smelting. Canals, and then railways, lowered the cost of moving raw materials and finished goods, so that markets which had once been local became national and then international. Cities grew quickly as people left the countryside to look for work in the new factories, and the resulting crowding created serious problems of housing, sanitation, and public health. At the same time, rising output made a wider range of goods available at lower prices, and over the long run living standards improved for most of the population. Historians still debate how quickly those gains reached ordinary workers, and how much of the early growth depended on colonial trade and on the exploitation of enslaved labor in the Atlantic economy. What is not in doubt is that the changes set in motion during this period reshaped politics, family life, education, and the relationship between people and the natural world in ways that are still unfolding today."""


TEXTS = {
    "storia": DEFAULT_TEXT,
    "codice": 'def merge_sorted(a, b):\n    """Merge two sorted lists into one sorted list."""\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    out.extend(a[i:])\n    out.extend(b[j:])\n    return out\n\n\nclass LRUCache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n        self.data = {}\n        self.order = []\n\n    def get(self, key):\n        if key not in self.data:\n            return None\n        self.order.remove(key)\n        self.order.append(key)\n        return self.data[key]\n\n    def put(self, key, value):\n        if key in self.data:\n            self.order.remove(key)\n        elif len(self.data) >= self.capacity:\n            oldest = self.order.pop(0)\n            del self.data[oldest]\n        self.data[key] = value\n        self.order.append(key)\n\n\nif __name__ == "__main__":\n    print(merge_sorted([1, 4, 9], [2, 3, 10]))\n    c = LRUCache(2)\n    c.put("a", 1); c.put("b", 2); c.get("a"); c.put("c", 3)\n    print(c.get("b"), c.get("a"), c.get("c"))\n',
    "dialogo": '"Are you sure this is the right road?" Maria asked, peering through the rain at the sign that had long since lost most of its paint.\n"The map says so," Tom replied, though he did not sound convinced. He turned the paper around twice before folding it and putting it back in his pocket.\n"The map also said there would be a bridge."\n"There was a bridge. It just wasn\'t there anymore."\nShe laughed despite herself. The car\'s heater rattled, and the windshield wipers kept a slow, uneven rhythm. Somewhere ahead, a light flickered between the trees.\n"Do you think that\'s the farmhouse?"\n"It\'s either the farmhouse or someone who very much wants us to think it is."\n"That\'s not reassuring."\n"It wasn\'t meant to be." He slowed down as the road narrowed into gravel. "Look, if it\'s the wrong place, we turn around. We\'ve done it three times tonight already. One more won\'t kill us."\n"Famous last words," she said, and pulled her coat tighter as the light grew closer and the rain, if anything, came down harder.',
    "scienza": 'Photosynthesis converts light energy into chemical energy stored in the bonds of sugar molecules. In plants, the process takes place in chloroplasts, where chlorophyll and other pigments absorb photons, mainly in the blue and red parts of the visible spectrum. The light-dependent reactions occur in the thylakoid membranes: absorbed energy drives the splitting of water, releasing oxygen as a by-product and producing ATP and NADPH. These energy carriers are then consumed in the Calvin cycle in the stroma, where the enzyme RuBisCO fixes carbon dioxide onto a five-carbon sugar, ribulose bisphosphate. The resulting six-carbon intermediate immediately splits into two three-carbon molecules, which are reduced and partly recycled to regenerate the acceptor, while the remainder is exported to build glucose, starch, and cellulose. The overall efficiency of the process is low, typically one to two percent of incident sunlight, because much of the energy is lost as heat, reflected, or spent on photorespiration when RuBisCO binds oxygen instead of carbon dioxide. C4 and CAM plants have evolved anatomical and biochemical adaptations that concentrate carbon dioxide around RuBisCO, reducing that loss in hot, dry environments.',
}


def run_one(tok, exe, text, attn):
    ids = tok(text)["input_ids"]
    env = dict(os.environ, AIM_ATTN=attn)
    r = subprocess.run([exe, os.environ.get("AIM_MODEL", "models/bitnet-2b-4t.aim"), "--ids", ",".join(map(str, ids)), "--ppl"],
                       capture_output=True, text=True, env=env)
    return float(r.stdout.split()[-1]), len(ids) - 1


def main():
    args = sys.argv[1:]
    attn = "softmax"; text = DEFAULT_TEXT
    if "--attn" in args:
        i = args.index("--attn"); attn = args[i + 1]; del args[i:i + 2]
    if "--text" in args:
        i = args.index("--text"); text = open(args[i + 1], encoding="utf-8").read(); del args[i:i + 2]
    tok = AutoTokenizer.from_pretrained(REPO)
    exe = os.path.join("build", "run.exe" if os.name == "nt" else "run")
    if "--all" in args:
        modes = ["softmax", "max", "topk=8", "topk=32", "h=0.85", "h=0.7"]
        print("%-8s %6s " % ("testo", "token") + " ".join("%9s" % m for m in modes))
        for name, t in TEXTS.items():
            row = [run_one(tok, exe, t, m) for m in modes]
            print("%-8s %6d " % (name, row[0][1]) + " ".join("%9.3f" % v for v, _ in row), flush=True)
        return
    ids = tok(text)["input_ids"]
    env = dict(os.environ, AIM_ATTN=attn)
    r = subprocess.run([exe, os.environ.get("AIM_MODEL", "models/bitnet-2b-4t.aim"), "--ids", ",".join(map(str, ids)), "--ppl"],
                       capture_output=True, text=True, env=env)
    print(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.stdout.strip())


if __name__ == "__main__":
    main()
