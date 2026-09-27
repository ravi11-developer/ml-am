"""Phase A2-A5: how were S2/S3 copies generated, and what makes matching hard?
    python pattern_analysis.py > artifacts/pattern_analysis.txt
Reads raw train TSVs + ground truth only (plus artifacts/D1/cache for the FP owner check)."""
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "code/business_entity_resolution/src")
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from rapidfuzz.distance import Levenshtein

from er_pipeline import read_tsv, col, normalize, ABBREV, LEGAL, _translit_indic, skeleton

D = "dataset/train/"
rng = np.random.default_rng(0)


def load(i):
    t = read_tsv(f"{D}train_source{i}.tsv")
    return {c: col(t, c) for c in ("entity_id", "business_name", "business_address", "country")}


S1, S2, S3 = load(1), load(2), load(3)
n1, n2, n3 = len(S1["entity_id"]), len(S2["entity_id"]), len(S3["entity_id"])
T = {c: pa.concat_arrays([S2[c], S3[c]]) for c in S1}
print(f"S1 {n1:,}  S2 {n2:,}  S3 {n3:,}")

g = read_tsv(f"{D}train_ground_truth.tsv")
L = pc.split_pattern(col(g, "matched_entity_ids"), ",")
par = pc.list_parent_indices(L).to_numpy()
flat = pc.utf8_trim_whitespace(pc.list_flatten(L))
ne = pc.not_equal(flat, "").to_numpy(zero_copy_only=False)
gq = pc.index_in(pc.utf8_trim_whitespace(col(g, "source1_entity_id")), value_set=pc.utf8_trim_whitespace(S1["entity_id"])).to_numpy(zero_copy_only=False)
gt = pc.index_in(flat, value_set=pc.utf8_trim_whitespace(T["entity_id"])).to_numpy(zero_copy_only=False)
q_of = gq[par]
ok = ne & ~np.isnan(gt.astype(float)) & ~np.isnan(q_of.astype(float))
PQ, PT = q_of[ok].astype(np.int64), gt[ok].astype(np.int64)
print(f"true pairs {len(PQ):,}")

# ---------------- A5: count structure ----------------
print("\n### A5 copies per S1 entity (S2 count, S3 count)")
c2 = np.bincount(PQ[PT < n2], minlength=n1)
c3 = np.bincount(PQ[PT >= n2], minlength=n1)
pairs = pd.Series(list(zip(c2, c3))).value_counts(normalize=True).head(15)
print((100 * pairs).round(2).to_string())
nz = (c2 + c3) > 0
print(f"matched S1 with >=1 copy in BOTH sources: {100 * ((c2 > 0) & (c3 > 0))[nz].mean():.1f}%   "
      f"only S2: {100 * ((c2 > 0) & (c3 == 0))[nz].mean():.1f}%   only S3: {100 * ((c2 == 0) & (c3 > 0))[nz].mean():.1f}%")
print(f"total copies per matched S1: " + ", ".join(f"{k}:{v:.1f}%" for k, v in
      (100 * pd.Series((c2 + c3)[nz]).value_counts(normalize=True).sort_index()).items()))
own = np.zeros(n2 + n3, bool); own[PT] = True
print(f"S2 records owned by some S1: {100 * own[:n2].mean():.1f}%   S3: {100 * own[n2:].mean():.1f}%")

# ---------------- canonical keys ----------------
LEGALX = LEGAL | {"pvt", "ltd", "inc", "corp", "co", "partners", "services", "service", "center", "centre", "group",
                  "enterprises", "enterprise", "associates", "llp", "lp", "pllc", "commission", "trust", "foundation",
                  "shri", "sri", "smt", "m", "s", "ms", "the"}
LEET = str.maketrans("01345678", "oleasgtb")


def canon_name_tokens(norm):
    out = []
    for w in norm.split():
        if any(ch.isdigit() for ch in w) and sum(ch.isalpha() for ch in w) >= 2:
            w = w.translate(LEET)
        w = ABBREV.get(w, w)
        if w and w not in LEGALX and w not in ("com", "www"):
            out.append(w)
    return out


def arr_norm(a, addr=False):
    return normalize(_translit_indic(a), is_addr=addr).to_pylist()


# ---------------- A3: duplicate names among S1 ----------------
s1_norm = arr_norm(S1["business_name"])
s1_key = [" ".join(sorted(set(canon_name_tokens(x)))) for x in s1_norm]
kc = Counter(s1_key)
shared = np.array([kc[k] > 1 for k in s1_key])
print("\n### A3 S1 entities sharing a canonical name with another S1 entity")
print(f"  share of S1 with a same-name sibling: {100 * shared.mean():.1f}%")
sizes = pd.Series(list(kc.values()))
print("  sibling-group size distribution (groups): " + ", ".join(
    f"{k}:{v}" for k, v in sizes.value_counts().sort_index().head(10).items()))
print("  most common canonical names: " + ", ".join(f"{k!r}x{v}" for k, v in kc.most_common(12)))

# ---------------- A2: transformations on true pairs ----------------
samp_q = rng.choice(np.flatnonzero(nz), 6000, replace=False)
m = np.isin(PQ, samp_q)
sq, st = PQ[m], PT[m]
print(f"\n### A2 transformations on {len(sq):,} true pairs of 6,000 random matched S1 entities")
take = lambda a, i: a.take(pa.array(i)).to_pylist()
rn1, rn2 = take(S1["business_name"], sq), take(T["business_name"], st)
ra1, ra2 = take(S1["business_address"], sq), take(T["business_address"], st)
nn1 = arr_norm(pa.array(rn1, pa.large_string())); nn2 = arr_norm(pa.array(rn2, pa.large_string()))
na1 = arr_norm(pa.array(ra1, pa.large_string()), True); na2 = arr_norm(pa.array(ra2, pa.large_string()), True)
NATIVE = re.compile(r"[ऀ-෿]")
ALIAS = re.compile(r"\b(dba|d/b/a|aka|a/k/a|fka|f/k/a|formerly|doing business as|trading as|formerly known as)\b", re.I)
DOMAIN = re.compile(r"(\.\s*(com|net|org|in|co)\b|^\s*www\.|^\s*#)", re.I)
LEETRE = re.compile(r"[A-Za-z][0-9][A-Za-z]|\b[0-9][A-Za-z]{2,}|[A-Za-z]{2,}[0-9]\b")
DUP = re.compile(r"\b(\w+)[ ,]+\1\b", re.I)
lab = Counter()
key_eq = Counter()
examples = {}
for i in range(len(sq)):
    a, b = rn1[i] or "", rn2[i] or ""
    A, B = nn1[i], nn2[i]
    ta, tb = canon_name_tokens(A), canon_name_tokens(B)
    labels = []
    if a == b: labels.append("name: identical raw")
    elif a.lower().split() == b.lower().split(): labels.append("name: case/space only")
    if NATIVE.search(b): labels.append("name: native script")
    if ALIAS.search(b): labels.append("name: alias marker")
    if DOMAIN.search(b) or (" " not in b.strip() and len(b) > 8 and " " in a.strip()): labels.append("name: domain/handle/concatenated")
    if LEETRE.search(b) and not LEETRE.search(a): labels.append("name: leet digits")
    if DUP.search(b) and not DUP.search(a): labels.append("name: duplicated token")
    if A != B and sorted(A.split()) == sorted(B.split()): labels.append("name: token reorder")
    if set(ta) == set(tb) and set(A.split()) != set(B.split()): labels.append("name: legal/prefix/suffix change only")
    if set(ta) and set(tb) and set(tb) < set(ta): labels.append("name: core tokens dropped")
    if set(ta) and set(tb) and set(ta) < set(tb): labels.append("name: core tokens added")
    if not (set(ta) & set(tb)) and not NATIVE.search(b):
        labels.append("name: NO shared core token (Latin)")
    if set(ta) != set(tb) and Levenshtein.normalized_similarity(" ".join(sorted(ta)), " ".join(sorted(tb))) >= 0.8:
        labels.append("name: small typo in core")
    # canonical-key equalities
    k_name = " ".join(sorted(set(ta))) == " ".join(sorted(set(tb))) and bool(ta)
    k_skel = sorted({skeleton(w) for w in ta} - {""}) == sorted({skeleton(w) for w in tb} - {""}) and bool(ta)
    ua, ub = set(na1[i].split()), set(na2[i].split())
    nums_a = {w for w in ua if w.isdigit()}; nums_b = {w for w in ub if w.isdigit()}
    if not ub: labels.append("addr: target empty")
    elif ua == ub: labels.append("addr: same token set")
    else:
        if ub < ua: labels.append("addr: target subset (components dropped)")
        if ua < ub: labels.append("addr: target superset (tokens added)")
        if len(ua & ub) / len(ua | ub) >= 0.5: labels.append("addr: jaccard>=0.5")
        if nums_a and nums_b and not (nums_a & nums_b): labels.append("addr: numbers all differ")
        if any(x != y and (x.endswith(y) or y.endswith(x) or x.startswith(y) or y.startswith(x) or x.lstrip("0") == y.lstrip("0"))
               for x in nums_a for y in nums_b) and not (nums_a & nums_b):
            labels.append("addr: house number truncated/padded")
    k_addr = bool(ub) and (ub <= ua or ua <= ub or len(ua & ub) / len(ua | ub) >= 0.6)
    for l in labels:
        lab[l] += 1
        examples.setdefault(l, []).append(i)
    key_eq["canonical name set equal"] += k_name
    key_eq["name skeleton set equal"] += k_skel
    key_eq["address subset/superset or jaccard>=0.6"] += k_addr
    key_eq["name-set equal OR skeleton equal"] += (k_name or k_skel)
    key_eq["(name or skeleton) AND address"] += ((k_name or k_skel) and k_addr)
    key_eq["(name or skeleton) OR address"] += ((k_name or k_skel) or k_addr)
N = len(sq)
for l, c in sorted(lab.items()):
    print(f"  {l:48s} {100 * c / N:5.1f}%")
print("  canonical-key agreement on true pairs:")
for l, c in key_eq.items():
    print(f"    {l:48s} {100 * c / N:5.1f}%")
print("\n  examples:")
for l in sorted(examples):
    for i in examples[l][:2]:
        print(f"   [{l}] {rn1[i]} | {ra1[i]}  ->  {rn2[i]} | {ra2[i]}")

# ---------------- A4: validation FPs of D1 - are they sibling confusions? ----------------
V = pd.read_parquet("artifacts/D1/cache/val_scored.parquet")
import json
thr = json.loads(Path("artifacts/D1/run_report.json").read_text())["validation"]["threshold"]
sys.path.insert(0, "code/business_entity_resolution/src")
from er_pipeline import target_best
tb = target_best(V["t"].to_numpy(), V["prob"].to_numpy())
fp = V[(V["prob"] >= thr) & tb & (V["label"] == 0)]
owner = np.full(n2 + n3, -1, np.int64); owner[PT] = PQ
o = owner[fp["t"].to_numpy()]
has_o = o >= 0
same_name = np.array([s1_key[q] == s1_key[w] for q, w in zip(fp["q"].to_numpy()[has_o], o[has_o])])
print(f"\n### A4 D1 validation false positives: {len(fp):,}")
print(f"  target owned by another S1: {100 * has_o.mean():.1f}%;  of those, owner has the SAME canonical name "
      f"as the wrongly matched S1: {100 * same_name.mean():.1f}%")
print(f"  target owned by nobody (decoy): {100 * (~has_o).mean():.1f}%")
for (qq, tt, pp), w in list(zip(fp[["q", "t", "prob"]].itertuples(index=False), o))[:12]:
    print(f"   p={pp:.2f} S1: {S1['business_name'][int(qq)].as_py()} | {S1['business_address'][int(qq)].as_py()}")
    print(f"          T : {T['business_name'][int(tt)].as_py()} | {T['business_address'][int(tt)].as_py()}")
    if w >= 0:
        print(f"      OWNER : {S1['business_name'][int(w)].as_py()} | {S1['business_address'][int(w)].as_py()}")
# decoys: what do unowned records look like?
un = rng.choice(np.flatnonzero(~own), 8, replace=False)
print("\n  random unowned (decoy) records:")
for u in un:
    print(f"   {T['business_name'][int(u)].as_py()} | {T['business_address'][int(u)].as_py()} | {T['country'][int(u)].as_py()}")
