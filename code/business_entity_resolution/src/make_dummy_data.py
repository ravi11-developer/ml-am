"""SYNTHETIC data in the challenge layout, for smoke tests / timing only (never for training).
    python src/make_dummy_data.py --out dummy_dataset --n 50000
"""
import argparse, random
from pathlib import Path
import numpy as np

SYL = ["ka", "ra", "mo", "li", "sun", "te", "va", "shi", "an", "go", "pa", "del", "ber", "mi", "no", "tor",
       "ja", "lu", "ven", "sa", "ri", "ko", "dra", "el", "ha", "zu", "pre", "qui", "lo", "ne"]
KINDS = ["traders", "foods", "motors", "textiles", "pharma", "logistics", "bakery", "consulting", "electronics",
         "hardware", "jewellers", "exports", "salon", "dental", "boulangerie", "restaurant", "clinic", "studio"]
LEGAL = {"US": ["Inc", "LLC", "Corp", "Corporation", "Co"], "India": ["Pvt Ltd", "Private Limited", "LLP", "Enterprises"],
         "France": ["SARL", "SAS", "SA", "EURL"]}
STREET = {"US": ["St", "Street", "Ave", "Road", "Blvd"], "India": ["Road", "Nagar", "Marg", "Rd", "Colony"],
          "France": ["Rue", "Avenue", "Boulevard", "Place"]}
STATES = {"US": ["TX", "CA", "NY", "FL", "IL"], "India": ["Maharashtra", "Karnataka", "Tamil Nadu", "Delhi"],
          "France": ["Île-de-France", "Rhône", "Gironde"]}


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed); rs = np.random.default_rng(seed)
        self.words = list({"".join(self.r.choice(SYL) for _ in range(self.r.randint(2, 3))).title() for _ in range(40000)})
        z = 1 / np.arange(1, len(self.words) + 1) ** 1.05; self.p = z / z.sum()
        self.rs = rs
        self.cities = {c: [self.word() for _ in range(400)] for c in STATES}

    def word(self):
        return self.words[self.rs.choice(len(self.words), p=self.p)]

    def entity(self, c):
        r = self.r
        name = f"{self.word()} {self.word() if r.random() < 0.7 else ''} {r.choice(KINDS).title()} {r.choice(LEGAL[c])}"
        street = f"{self.word()} {r.choice(STREET[c])}" if c != "France" else f"{r.choice(STREET[c])} {self.word()}"
        zl = 6 if c == "India" else 5
        addr = f"{r.randint(1, 3000)} {street}, {r.choice(self.cities[c])}, {r.choice(STATES[c])} {r.randint(10**(zl-1), 10**zl-1)}"
        return " ".join(name.split()), addr, c

    def noisy(self, rec):
        r = self.r; name, addr, c = rec
        sw = {"Street": "St", "St": "Street", "Road": "Rd", "Rd": "Road", "Private": "Pvt", "Limited": "Ltd",
              "Corporation": "Corp", "Avenue": "Ave", "Boulevard": "Bd"}
        t = [sw.get(x, x) if r.random() < .4 else x for x in name.split()]
        if r.random() < .3 and len(t) > 2: t = t[:-1]
        if r.random() < .35:
            i = r.randrange(len(t)); w = t[i]
            if len(w) > 3: j = r.randrange(len(w) - 1); t[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        if r.random() < .1 and len(t) > 2: t[0], t[1] = t[1], t[0]
        a = " ".join(sw.get(x, x) if r.random() < .4 else x for x in addr.replace(",", " ,").split()).replace(" ,", ",")
        if r.random() < .3: a = a.rsplit(" ", 1)[0]
        if r.random() < .15: a = a.split(" ", 1)[1]
        if r.random() < .15 and c == "India": a = "Near SBI ATM, " + a
        if r.random() < .05: a = ""
        return (" ".join(t).upper() if r.random() < .2 else " ".join(t)), a, c


def build(g, split, n, countries, out):
    r = g.r; d = out / split; d.mkdir(parents=True, exist_ok=True)
    s1, s2, s3, gt = [], [], [], []
    for k in range(n):
        base = g.entity(r.choice(countries)); sid = f"S1-{k:07d}"; s1.append((sid, *base)); m = []
        if r.random() > 0.25:
            for src, lst in (("S2", s2), ("S3", s3)):
                for _ in range(r.choice([0, 1, 1, 2, 2, 3])):
                    tid = f"{src}-{len(lst):08d}"; lst.append((tid, *g.noisy(base))); m.append(tid)
        gt.append((sid, ",".join(m)))
    for src, lst in (("S2", s2), ("S3", s3)):
        for _ in range(n // 2):
            lst.append((f"{src}-{len(lst):08d}", *g.noisy(g.entity(r.choice(countries)))))
    r.shuffle(s2); r.shuffle(s3)
    h = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    for fn, rows in ((f"{split}_source1.tsv", s1), (f"{split}_source2.tsv", s2), (f"{split}_source3.tsv", s3)):
        (d / fn).write_text(h + "".join("\t".join(x) + "\n" for x in rows), encoding="utf-8")
    return gt


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="dummy_dataset"); ap.add_argument("--n", type=int, default=50000)
    a = ap.parse_args(); out = Path(a.out)
    g = Gen(0)
    gt = build(g, "train", a.n, ["US", "India"], out)
    (out / "train" / "train_ground_truth.tsv").write_text("source1_entity_id\tmatched_entity_ids\n" + "".join(f"{x}\t{y}\n" for x, y in gt))
    gt = build(g, "test", a.n, ["US", "India", "France"], out)
    (out / "test_ground_truth_HIDDEN.tsv").write_text("source1_entity_id\tmatched_entity_ids\n" + "".join(f"{x}\t{y}\n" for x, y in gt))
    print("written", out)
