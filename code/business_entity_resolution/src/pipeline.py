"""Business Entity Resolution pipeline (same code as notebooks/er_pipeline.ipynb).
Run:  ER_BASE=/path/to/student_resource python pipeline.py
"""
# ==== Cell 2: Imports, paths and configuration ====
import os, re, sys, csv, json, time, pickle, random, subprocess, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from unidecode import unidecode
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
import jellyfish
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)

# ---- Paths: change BASE if your folder lives elsewhere ----
BASE = Path(os.environ.get("ER_BASE", "/home/sagemaker-user/student_resource"))
TRAIN_DIR = BASE / "dataset" / "train"
TEST_DIR = BASE / "dataset" / "test"
OUT_DIR = BASE / "output"
ART_DIR = BASE / "artifacts"
VALIDATOR = BASE / "utils" / "validate_submission.py"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED = ([TRAIN_DIR / f"train_source{k}.tsv" for k in (1, 2, 3)]
            + [TRAIN_DIR / "train_ground_truth.tsv"]
            + [TEST_DIR / f"test_source{k}.tsv" for k in (1, 2, 3)])
missing = [str(p) for p in EXPECTED if not p.exists()]
if missing:
    raise FileNotFoundError("Missing files:\n  " + "\n  ".join(missing) + f"\nCheck BASE = {BASE}")
print("All data files found under", BASE)

CFG = dict(
    seed=42,
    k_name=30, k_addr=20, k_full=30, k_rev=5,  # blocking: top-K per method
    postal_max_group=50,      # postal+token blocking: skip keys shared by more records than this
    block_by_country=True,    # only compare records with the same country label (check EDA)
    one_home=True,            # each S2/S3 record goes to at most one S1 (check EDA)
    n_folds=5,
    beta=0.5,
)
random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
T0 = time.time()


def tick(msg):
    """Print a message with elapsed time since the notebook started."""
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)

# ==== Cell 3: Load data ====
def read_tsv(path):
    """Read a challenge TSV with every column as string.
    QUOTE_NONE stops a stray quote character in a name from swallowing following lines."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    df.columns = [c.strip().strip('"') for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip().str.strip('"')
    return df


def load_split(split):
    """Return (source1, others) for 'train' or 'test'. 'others' = Source 2 + Source 3 with a 'source' column."""
    d = TRAIN_DIR if split == "train" else TEST_DIR
    s1 = read_tsv(d / f"{split}_source1.tsv")
    others = pd.concat([read_tsv(d / f"{split}_source2.tsv"),
                        read_tsv(d / f"{split}_source3.tsv")], ignore_index=True)
    for df in (s1, others):
        for col in ("business_name", "business_address", "country"):
            if col not in df.columns:
                df[col] = ""
        # country is an OPEN set of labels: we only lower-case it, never filter to known values
        df["ckey"] = df["country"].str.strip().str.lower().replace("", "unknown")
    others["source"] = others["entity_id"].str[:2]
    return s1.reset_index(drop=True), others.reset_index(drop=True)


def load_truth(s1_ids):
    """Ground truth as {s1_id: set(matched ids)}; every S1 id is present (empty set = singleton)."""
    gt = read_tsv(TRAIN_DIR / "train_ground_truth.tsv")
    truth = {sid: set() for sid in s1_ids}
    for sid, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        truth.setdefault(sid, set()).update(x.strip() for x in m.split(",") if x.strip())
    return truth, gt


tr_s1, tr_ot = load_split("train")
truth, gt_raw = load_truth(tr_s1["entity_id"])
true_pairs = [(s, m) for s, ms in truth.items() for m in ms]     # list of (S1 id, matched S2/S3 id)
tick(f"train: S1={len(tr_s1):,}  S2+S3={len(tr_ot):,}  ground-truth rows={len(gt_raw):,}")
print(tr_s1.head(3).to_string())
print(tr_ot.head(3).to_string())

# ==== Cell 4: Exploratory checks (read these numbers before tuning anything) ====
te_s1_raw, te_ot_raw = load_split("test")


def show_counts(s1, ot, name):
    """Record counts per country and source."""
    t = (pd.concat([s1.assign(src="S1"), ot.assign(src=ot["source"])])
         .groupby(["country", "src"]).size().unstack(fill_value=0))
    print(f"\n{name} record counts:\n{t}")


show_counts(tr_s1, tr_ot, "TRAIN")
show_counts(te_s1_raw, te_ot_raw, "TEST")

n_match = pd.Series({k: len(v) for k, v in truth.items()})
print(f"\nSingleton rate in train: {(n_match == 0).mean():.1%}")
print("Matches per S1 entity:", n_match.value_counts().sort_index().head(12).to_dict())

# Is each S2/S3 record matched to at most one S1 entity? (decides CFG['one_home'])
seen, multi = set(), 0
for ms in truth.values():
    for m in ms:
        multi += m in seen
        seen.add(m)
print(f"\nS2/S3 ids appearing under more than one S1: {multi} -> one_home is "
      f"{'SAFE' if multi == 0 else 'NOT safe: set CFG[one_home]=False'}")

c1 = dict(zip(tr_s1.entity_id, tr_s1.ckey))
c2 = dict(zip(tr_ot.entity_id, tr_ot.ckey))
mis = sum(c1.get(s) != c2.get(m) for s, m in true_pairs)
print(f"True pairs: {len(true_pairs):,}; country label mismatch: {mis} ({mis / max(1, len(true_pairs)):.2%})"
      f" -> if this is above ~1%, set CFG['block_by_country']=False")
print("True matches by source:", pd.Series([m[:2] for _, m in true_pairs]).value_counts().to_dict())

print("\nEmpty-field rates:")
for nm, df in (("S1", tr_s1), ("S2+S3", tr_ot)):
    print(f"  {nm}:", {c: f"{(df[c] == '').mean():.1%}" for c in ("business_name", "business_address", "country")})

recs = pd.concat([tr_s1, tr_ot]).set_index("entity_id")
print("\n10 random true pairs (S1 | match):")
for s, m in random.Random(1).sample(true_pairs, min(10, len(true_pairs))):
    print(f"  {recs.at[s, 'business_name']} | {recs.at[s, 'business_address']}")
    print(f"  {recs.at[m, 'business_name']} | {recs.at[m, 'business_address']}\n")

# ==== Cell 5: Text normalization ====
# All rules below are hand-written string rules (no external data). Ambiguous abbreviations are mapped to one
# shared short form (e.g. street/st/saint -> "st") so both sides always normalize the same way.

LEGAL = {  # legal-form variants -> canonical; removed from the "core" name and kept as a separate feature
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "cie": "co", "compagnie": "co",
    "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt", "pte": "pvt",
    "llc": "llc", "llp": "llp", "plc": "plc", "lp": "lp", "pllc": "pllc", "opc": "opc", "gmbh": "gmbh",
    "sa": "sa", "sas": "sas", "sasu": "sasu", "sarl": "sarl", "eurl": "eurl", "snc": "snc", "sci": "sci",
    "selarl": "selarl", "scop": "scop",
}
NAME_CANON = {  # common business-word abbreviations -> one form
    "technologies": "tech", "technology": "tech", "techno": "tech",
    "enterprises": "ent", "enterprise": "ent", "services": "svc", "service": "svc", "svcs": "svc",
    "international": "intl", "natl": "natl", "national": "natl", "manufacturing": "mfg",
    "management": "mgmt", "associates": "assoc", "brothers": "bros", "industries": "inds",
    "group": "grp", "holdings": "hldgs", "solutions": "sol", "systems": "sys",
    "pharmaceuticals": "pharma", "pharmaceutical": "pharma", "center": "ctr", "centre": "ctr",
    "etablissements": "ets", "university": "univ", "hospital": "hosp", "saint": "st",
}
NAME_STOP = {"the", "and", "of", "et", "de", "la", "le", "les", "du", "des", "l", "d", "s"}

_ADDR_GROUPS = {
    "rd": ["road", "rd"], "st": ["street", "st", "str", "saint"], "ste": ["suite", "ste", "sainte"],
    "ave": ["avenue", "ave", "av"], "blvd": ["boulevard", "blvd", "bd", "boul"], "ln": ["lane", "ln"],
    "dr": ["drive", "dr"], "ct": ["court", "ct"], "pl": ["place", "pl"], "sq": ["square", "sq"],
    "hwy": ["highway", "hwy"], "pkwy": ["parkway", "pkwy"], "expy": ["expressway", "expy"],
    "cir": ["circle", "cir"], "jn": ["junction", "jn", "jct", "jnc"], "apt": ["apartment", "apartments", "apt", "apts"],
    "bldg": ["building", "bldg", "bld"], "fl": ["floor", "fl", "flr"], "nr": ["near", "nr"],
    "opp": ["opposite", "opp", "opps"], "bhd": ["behind", "bhd"], "ngr": ["nagar", "ngr"],
    "sec": ["sector", "sec", "sect"], "mkt": ["market", "mkt"], "stn": ["station", "stn"],
    "clny": ["colony", "clny"], "extn": ["extension", "extn", "ext"], "cmplx": ["complex", "cmplx", "complx"],
    "chk": ["chowk", "chk"], "no": ["number", "no", "num", "nbr"], "n": ["north", "n"], "s": ["south", "s"],
    "e": ["east", "e"], "w": ["west", "w"], "mt": ["mount", "mt"], "ft": ["fort", "ft"], "rte": ["route", "rte"],
    "imp": ["impasse", "imp"], "fbg": ["faubourg", "fbg"], "che": ["chemin", "che", "chem"],
    "twr": ["tower", "twr"], "ctr": ["center", "centre", "ctr", "cntr"], "plz": ["plaza", "plz"],
    "hts": ["heights", "hts"], "pk": ["park", "pk"], "crs": ["cross", "crs"],
}
ADDR_CANON = {v: k for k, vs in _ADDR_GROUPS.items() for v in vs}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi", "wyoming": "wy",
}
MULTI_WORD = {  # phrase replacements applied before token mapping
    "new york": "ny", "new jersey": "nj", "new mexico": "nm", "new hampshire": "nh", "north carolina": "nc",
    "north dakota": "nd", "south carolina": "sc", "south dakota": "sd", "west virginia": "wv",
    "rhode island": "ri", "district of columbia": "dc", "new delhi": "delhi",
}
CITY_ALIAS = {  # historical / transliterated city spellings -> one form
    "bangalore": "bengaluru", "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai",
    "gurgaon": "gurugram", "poona": "pune", "baroda": "vadodara", "trivandrum": "thiruvananthapuram",
    "mysore": "mysuru", "mangalore": "mangaluru", "vizag": "visakhapatnam", "cochin": "kochi",
    "pondicherry": "puducherry",
}
LANDMARK = {"nr", "opp", "bhd", "beside", "next", "adjacent", "landmark", "pres", "face"}
_MULTI_RE = re.compile(r"\b(" + "|".join(map(re.escape, MULTI_WORD)) + r")\b")
_PIN_SPLIT = re.compile(r"\b(\d{3}) (\d{3})\b")      # "560 001" -> "560001"
_POSTAL = re.compile(r"^\d{5,6}$")                    # US ZIP / India PIN / France code postal


def base_clean(s):
    """Lowercase, strip accents, unify punctuation. 'P.V.T.' -> 'pvt', '&' -> 'and'."""
    s = unidecode(str(s)).lower()
    s = re.sub(r"(?<=[a-z])\.(?=[a-z])", "", s)       # dots inside abbreviations
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def normalize_name(raw):
    """Return (core_name, legal_suffixes). Legal words are removed unless they are the first token."""
    toks = base_clean(raw).split()
    core, legal = [], set()
    for k, t in enumerate(toks):
        if t == "pvtltd":
            legal.update(["pvt", "ltd"])
        elif t in LEGAL and k > 0:
            legal.add(LEGAL[t])
        elif t in NAME_STOP:
            continue
        else:
            core.append(NAME_CANON.get(t, t))
    if not core:
        core = [NAME_CANON.get(t, t) for t in toks]
    return " ".join(core), " ".join(sorted(legal))


def normalize_address(raw):
    """Return (normalized address, postal code, other numbers, first number, has_landmark)."""
    s = base_clean(raw)
    s = _PIN_SPLIT.sub(r"\1\2", s)
    s = _MULTI_RE.sub(lambda m: MULTI_WORD[m.group(1)], s)
    toks = []
    for t in s.split():
        t = CITY_ALIAS.get(t, t)
        t = US_STATES.get(t, t)
        toks.append(ADDR_CANON.get(t, t))
    pos = [k for k, t in enumerate(toks) if _POSTAL.match(t)]
    postal = ""
    if pos and not (pos[-1] == 0 and len(toks) > 1):   # a lone 5-digit number at the start is a house number
        postal = toks[pos[-1]]
    nums = [t for t in toks if any(ch.isdigit() for ch in t) and t != postal]
    landmark = int(any(t in LANDMARK for t in toks))
    return " ".join(toks), postal, " ".join(sorted(set(nums))), (nums[0] if nums else ""), landmark


def acronym(core):
    """'tata consultancy svc' -> 'tcs' (only for names with 2+ tokens)."""
    toks = core.split()
    return "".join(t[0] for t in toks) if len(toks) >= 2 else ""


def add_norm_columns(df):
    """Add all normalized columns used by blocking and features."""
    df = df.copy()
    nm = [normalize_name(x) for x in df["business_name"]]
    df["name_core"] = [a for a, _ in nm]
    df["legal"] = [b for _, b in nm]
    df["name_clean"] = [base_clean(x) for x in df["business_name"]]
    df["name_compact"] = df["name_core"].str.replace(" ", "", regex=False)
    df["name_acr"] = [acronym(x) for x in df["name_core"]]
    df["name_first"] = [x.split()[0] if x else "" for x in df["name_core"]]
    df["name_phon"] = [" ".join(jellyfish.metaphone(t) for t in x.split() if t.isalpha()) for x in df["name_core"]]
    ad = [normalize_address(x) for x in df["business_address"]]
    df["addr_norm"] = [a[0] for a in ad]
    df["postal"] = [a[1] for a in ad]
    df["nums"] = [a[2] for a in ad]
    df["first_num"] = [a[3] for a in ad]
    df["landmark"] = [a[4] for a in ad]
    return df


tr_s1 = add_norm_columns(tr_s1)
tr_ot = add_norm_columns(tr_ot)
tick("normalized train")
print(tr_ot[["business_name", "name_core", "legal", "business_address", "addr_norm", "postal"]].head(8).to_string())

# ==== Cell 6: TF-IDF vectors and blocking (candidate generation) ====
class SplitText:
    """TF-IDF matrices for one split (train or test), fitted on that split's own text (no labels used).
    N = name char n-grams, A = address char n-grams, W = name word tokens, F = name+address combined."""

    def __init__(self, s1, ot):
        def fit(col, **kw):
            v = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, **kw)
            v.fit(pd.concat([s1[col], ot[col]]))
            return v.transform(s1[col]).tocsr(), v.transform(ot[col]).tocsr()

        self.N1, self.N2 = fit("name_core", analyzer="char_wb", ngram_range=(3, 4))
        self.A1, self.A2 = fit("addr_norm", analyzer="char_wb", ngram_range=(3, 4))
        self.W1, self.W2 = fit("name_core", analyzer="word", token_pattern=r"(?u)\b\w+\b")
        # dot product of these equals (name_cos + address_cos) / 2
        self.F1 = (sp.hstack([self.N1, self.A1]) / np.sqrt(2)).tocsr()
        self.F2 = (sp.hstack([self.N2, self.A2]) / np.sqrt(2)).tocsr()


def topk_sparse(Q, D, k, max_cells=2e7):
    """For every row of Q, the top-k rows of D by dot product. Returns (q_idx, d_idx, score).
    Works in chunks so the dense block never exceeds max_cells floats."""
    if Q.shape[0] == 0 or D.shape[0] == 0:
        return np.array([], int), np.array([], int), np.array([], np.float32)
    k = min(k, D.shape[0])
    DT = D.T.tocsr()
    step = max(1, int(max_cells // D.shape[0]))
    rows, cols, vals = [], [], []
    for st in range(0, Q.shape[0], step):
        S = (Q[st:st + step] @ DT).toarray()
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        sc = np.take_along_axis(S, idx, axis=1).ravel()
        r = np.repeat(np.arange(st, st + S.shape[0]), k)
        keep = sc > 0
        rows.append(r[keep]); cols.append(idx.ravel()[keep]); vals.append(sc[keep])
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def postal_token_pairs(s1, ot, i_idx, j_idx, max_group):
    """Pairs that share the same postal code AND at least one uncommon name token."""
    def long_table(df, idx):
        """One row per (record, name token) for records that have a postal code."""
        sub = df.iloc[idx]
        sub = sub[sub["postal"] != ""]
        toks = sub["name_core"].str.split()
        lens = toks.str.len().values if len(sub) else np.array([], int)
        if lens.sum() == 0:
            return pd.DataFrame(columns=["row", "tok", "postal"])
        return pd.DataFrame({"row": np.repeat(sub.index.values, lens),
                             "tok": np.concatenate([t for t in toks.values if t]),
                             "postal": np.repeat(sub["postal"].values, lens)})

    L, R = long_table(s1, i_idx), long_table(ot, j_idx)
    if len(L) == 0 or len(R) == 0:
        return np.array([], int), np.array([], int)
    df_tok = pd.concat([L["tok"], R["tok"]]).value_counts()
    common = set(df_tok[df_tok > max(20, 0.01 * (len(i_idx) + len(j_idx)))].index)
    L = L[~L["tok"].isin(common)].assign(key=lambda d: d["postal"] + "|" + d["tok"])
    R = R[~R["tok"].isin(common)].assign(key=lambda d: d["postal"] + "|" + d["tok"])
    big = R["key"].value_counts()
    R = R[R["key"].isin(big[big <= max_group].index)]
    m = L[["row", "key"]].merge(R[["row", "key"]], on="key", suffixes=("_i", "_j")).drop_duplicates(["row_i", "row_j"])
    return m["row_i"].values.astype(int), m["row_j"].values.astype(int)


def generate_candidates(s1, ot, tx, cfg=CFG):
    """Union of 5 blocking methods, run separately inside each country label.
    Returns DataFrame (i, j, m_name, m_addr, m_full, m_rev, m_postal) with i/j = row positions in s1/ot."""
    parts = []
    countries = sorted(s1["ckey"].unique())
    for c in countries:
        i_idx = np.where(s1["ckey"].values == c)[0]
        j_idx = np.where(ot["ckey"].values == c)[0] if cfg["block_by_country"] else np.arange(len(ot))
        if len(i_idx) == 0 or len(j_idx) == 0:
            continue

        def add(method, qi, dj):
            parts.append(pd.DataFrame({"i": qi, "j": dj, method: 1}))

        for method, M1, M2, k in (("m_name", tx.N1, tx.N2, cfg["k_name"]),
                                  ("m_addr", tx.A1, tx.A2, cfg["k_addr"]),
                                  ("m_full", tx.F1, tx.F2, cfg["k_full"])):
            q, d, _ = topk_sparse(M1[i_idx], M2[j_idx], k)
            add(method, i_idx[q], j_idx[d])
        q, d, _ = topk_sparse(tx.N2[j_idx], tx.N1[i_idx], cfg["k_rev"])   # reverse: each S2/S3 -> its best S1s
        add("m_rev", i_idx[d], j_idx[q])
        pi, pj = postal_token_pairs(s1, ot, i_idx, j_idx, cfg["postal_max_group"])
        add("m_postal", pi, pj)
    pairs = pd.concat(parts, ignore_index=True).fillna(0)
    meth = ["m_name", "m_addr", "m_full", "m_rev", "m_postal"]
    for m in meth:
        if m not in pairs:
            pairs[m] = 0
    pairs = pairs.groupby(["i", "j"], as_index=False)[meth].max()
    pairs[meth] = pairs[meth].astype(np.int8)
    pairs[["i", "j"]] = pairs[["i", "j"]].astype(np.int32)
    return pairs

# ==== Cell 7: Run blocking on train and measure recall ====
tx_tr = SplitText(tr_s1, tr_ot)
pairs_tr = generate_candidates(tr_s1, tr_ot, tx_tr)
tick(f"train candidate pairs: {len(pairs_tr):,}")

s1_pos = {e: k for k, e in enumerate(tr_s1["entity_id"])}
ot_pos = {e: k for k, e in enumerate(tr_ot["entity_id"])}
true_ij = {(s1_pos[s], ot_pos[m]) for s, m in true_pairs if s in s1_pos and m in ot_pos}
pairs_tr["label"] = [int((a, b) in true_ij) for a, b in zip(pairs_tr["i"], pairs_tr["j"])]


def blocking_report(pairs, n_true, s1, ot):
    """Recall of the candidate set overall, per method, per country, per source; candidates per S1."""
    found = pairs["label"].sum()
    print(f"Pair recall (recall ceiling): {found / max(1, n_true):.4f}   ({found:,} of {n_true:,} true pairs)")
    for m in ["m_name", "m_addr", "m_full", "m_rev", "m_postal"]:
        print(f"  {m:9s} alone: recall {pairs.loc[pairs[m] == 1, 'label'].sum() / max(1, n_true):.4f}"
              f"   pairs {int(pairs[m].sum()):,}")
    tp = pairs[pairs["label"] == 1]
    ctry_true = pd.Series([s1.at[a, "ckey"] for a, _ in true_ij]).value_counts()
    ctry_found = s1.loc[tp["i"], "ckey"].value_counts()
    print("  recall by country:", (ctry_found / ctry_true).round(4).to_dict())
    src_true = pd.Series([ot.at[b, "source"] for _, b in true_ij]).value_counts()
    src_found = ot.loc[tp["j"], "source"].value_counts()
    print("  recall by source :", (src_found / src_true).round(4).to_dict())
    per = pairs.groupby("i").size().reindex(range(len(s1)), fill_value=0)
    print(f"  candidates per S1: mean {per.mean():.1f}, median {per.median():.0f}, max {per.max()}")


blocking_report(pairs_tr, len(true_ij), tr_s1, tr_ot)
# If recall is below ~0.97, raise k_name / k_full in CFG and re-run cells 6-7.

# ==== Cell 8: Pair features ====
def rowwise_dot(X, Y, i, j, chunk=200_000):
    """Cosine similarity of row X[i] with row Y[j] for every pair (rows are already L2-normalized)."""
    out = np.empty(len(i), dtype=np.float32)
    for s in range(0, len(i), chunk):
        out[s:s + chunk] = np.asarray(X[i[s:s + chunk]].multiply(Y[j[s:s + chunk]]).sum(axis=1)).ravel()
    return out


def pair_sim(a, b, scorer, scale=100.0):
    """Element-wise string similarity for two equal-length lists, scaled to 0-1."""
    try:
        r = process.cpdist(list(a), list(b), scorer=scorer, workers=-1)
    except Exception:  # older rapidfuzz without cpdist
        r = np.array([scorer(x, y) for x, y in zip(a, b)])
    return (np.asarray(r, dtype=np.float32) / scale)


def set_jaccard(a, b):
    """Jaccard similarity of space-separated token sets (NaN if either side is empty)."""
    out = np.full(len(a), np.nan, dtype=np.float32)
    for k, (x, y) in enumerate(zip(a, b)):
        if x and y:
            sx, sy = set(x.split()), set(y.split())
            out[k] = len(sx & sy) / len(sx | sy)
    return out


def group_context(f, key, col, prefix):
    """Rank / gap-to-best / margin-over-best-other of `col` inside each group of `key`."""
    g = f.groupby(key)[col]
    f[f"{prefix}_rank"] = g.rank(ascending=False, method="min").astype(np.float32)
    f[f"{prefix}_gap"] = (f[col] - g.transform("max")).astype(np.float32)
    s = f[[key, col]].sort_values([key, col], ascending=[True, False])
    s["r"] = s.groupby(key).cumcount()
    top1 = s[s["r"] == 0].set_index(key)[col]
    top2 = s[s["r"] == 1].set_index(key)[col]
    t1 = f[key].map(top1).values
    t2 = f[key].map(top2).fillna(-1).values
    f[f"{prefix}_margin"] = (f[col].values - np.where(f[col].values >= t1, t2, t1)).astype(np.float32)


def build_features(pairs, s1, ot, tx):
    """One row of numeric features per (S1, candidate) pair. No country one-hot (France is unseen)."""
    i, j = pairs["i"].values, pairs["j"].values
    A, B = s1.iloc[i].reset_index(drop=True), ot.iloc[j].reset_index(drop=True)
    f = pairs.reset_index(drop=True).copy()

    # --- name similarity ---
    f["n_char_cos"] = rowwise_dot(tx.N1, tx.N2, i, j)
    f["n_word_cos"] = rowwise_dot(tx.W1, tx.W2, i, j)
    f["n_ratio"] = pair_sim(A.name_core, B.name_core, fuzz.ratio)
    f["n_tset"] = pair_sim(A.name_core, B.name_core, fuzz.token_set_ratio)
    f["n_tsort"] = pair_sim(A.name_core, B.name_core, fuzz.token_sort_ratio)
    f["n_partial"] = pair_sim(A.name_core, B.name_core, fuzz.partial_ratio)
    f["n_jw"] = pair_sim(A.name_core, B.name_core, JaroWinkler.normalized_similarity, scale=1.0)
    f["n_compact"] = pair_sim(A.name_compact, B.name_compact, fuzz.ratio)
    f["n_phon"] = pair_sim(A.name_phon, B.name_phon, fuzz.token_set_ratio)
    f["n_raw"] = pair_sim(A.name_clean, B.name_clean, fuzz.ratio)
    f["n_tok_jacc"] = set_jaccard(A.name_core.values, B.name_core.values)
    f["n_first_eq"] = (A.name_first.values == B.name_first.values).astype(np.int8)
    f["n_acronym"] = (((A.name_acr.values != "") & (A.name_acr.values == B.name_compact.values)) |
                      ((B.name_acr.values != "") & (B.name_acr.values == A.name_compact.values))).astype(np.int8)
    f["n_exact"] = (A.name_core.values == B.name_core.values).astype(np.int8)
    la, lb = A.name_core.str.len().values, B.name_core.str.len().values
    f["n_len_min"], f["n_len_diff"] = np.minimum(la, lb), np.abs(la - lb)
    f["legal_both"] = ((A.legal.values != "") & (B.legal.values != "")).astype(np.int8)
    f["legal_eq"] = ((A.legal.values == B.legal.values) & (A.legal.values != "")).astype(np.int8)
    f["legal_conflict"] = (f["legal_both"].values.astype(bool) & (A.legal.values != B.legal.values)).astype(np.int8)

    # --- address similarity ---
    f["a_char_cos"] = rowwise_dot(tx.A1, tx.A2, i, j)
    f["a_tset"] = pair_sim(A.addr_norm, B.addr_norm, fuzz.token_set_ratio)
    f["a_partial"] = pair_sim(A.addr_norm, B.addr_norm, fuzz.partial_ratio)
    f["a_ratio"] = pair_sim(A.addr_norm, B.addr_norm, fuzz.ratio)
    f["a_empty_any"] = ((A.addr_norm.values == "") | (B.addr_norm.values == "")).astype(np.int8)
    pa, pb = A.postal.values, B.postal.values
    both = (pa != "") & (pb != "")
    f["postal_both"] = both.astype(np.int8)
    f["postal_eq"] = (both & (pa == pb)).astype(np.int8)
    f["postal_conflict"] = (both & (pa != pb)).astype(np.int8)
    f["postal_pref3_eq"] = (both & (A.postal.str[:3].values == B.postal.str[:3].values)).astype(np.int8)
    f["postal_one_missing"] = ((pa != "") ^ (pb != "")).astype(np.int8)
    f["num_jacc"] = set_jaccard(A.nums.values, B.nums.values)
    fa, fb = A.first_num.values, B.first_num.values
    f["first_num_eq"] = ((fa != "") & (fa == fb)).astype(np.int8)
    f["first_num_conflict"] = ((fa != "") & (fb != "") & (fa != fb)).astype(np.int8)
    f["landmark_any"] = (A.landmark.values | B.landmark.values).astype(np.int8)

    # --- combined + context ---
    f["full_cos"] = (f["n_char_cos"] + f["a_char_cos"]) / 2
    f["is_s3"] = (B.source.values == "S3").astype(np.int8)
    f["n_methods"] = f[["m_name", "m_addr", "m_full", "m_rev", "m_postal"]].sum(axis=1).astype(np.int8)
    for col in ("full_cos", "n_char_cos", "a_char_cos", "n_tset"):
        group_context(f, "i", col, f"gi_{col}")      # how this candidate ranks among the S1's candidates
    for col in ("full_cos", "n_char_cos"):
        group_context(f, "j", col, f"gj_{col}")      # how this S1 ranks among the candidate's possible S1s
    f["n_cand_i"] = f.groupby("i")["j"].transform("size").astype(np.int32)
    f["n_cand_j"] = f.groupby("j")["i"].transform("size").astype(np.int32)
    return f


feat_tr = build_features(pairs_tr, tr_s1, tr_ot, tx_tr)
FEATS = [c for c in feat_tr.columns if c not in ("i", "j", "label")]
tick(f"train features: {feat_tr.shape}, positives: {feat_tr['label'].sum():,}")

# ==== Cell 9: Competition metric (macro F0.5 per S1 entity) ====
def f_beta_entity(pred, true, beta=0.5):
    """F-beta for one S1 entity. Singleton: 1.0 if pred empty, else 0.0."""
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r, b2 = tp / len(pred), tp / len(true), beta ** 2
    return (1 + b2) * p * r / (b2 * p + r)


def macro_f(pred, truth_dict, ids=None, beta=0.5):
    """Average F-beta over `ids` (default: all S1 entities in truth). Missing predictions count as empty."""
    ids = list(truth_dict) if ids is None else list(ids)
    return float(np.mean([f_beta_entity(set(pred.get(s, ())), truth_dict[s], beta) for s in ids]))


def score_report(pred, truth_dict, ids=None, label=""):
    """Overall, singleton-only and non-singleton scores."""
    ids = list(truth_dict) if ids is None else list(ids)
    single = [s for s in ids if not truth_dict[s]]
    multi_ = [s for s in ids if truth_dict[s]]
    print(f"{label} macro F0.5 = {macro_f(pred, truth_dict, ids):.4f}   "
          f"singletons {macro_f(pred, truth_dict, single) if single else float('nan'):.4f} (n={len(single)})   "
          f"non-singletons {macro_f(pred, truth_dict, multi_) if multi_ else float('nan'):.4f} (n={len(multi_)})")


# unit test from the problem statement: expected 0.714
assert abs(f_beta_entity({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"}) - 0.714) < 1e-3
print("metric OK")
score_report({}, truth, label="Baseline 'predict nothing':")

# ==== Cell 10: LightGBM with GroupKFold (grouped by S1 entity) ====
LGB_PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.05, num_leaves=63,
                  min_child_samples=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, seed=CFG["seed"], verbose=-1, n_jobs=-1)

X, y, groups = feat_tr[FEATS], feat_tr["label"].values, feat_tr["i"].values
oof = np.zeros(len(feat_tr), dtype=np.float64)
models, best_iters = [], []
for fold, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=CFG["n_folds"]).split(X, y, groups)):
    dtr = lgb.Dataset(X.iloc[tr_idx], y[tr_idx])
    dva = lgb.Dataset(X.iloc[va_idx], y[va_idx])
    m = lgb.train(LGB_PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va_idx] = m.predict(X.iloc[va_idx], num_iteration=m.best_iteration)
    models.append(m)
    best_iters.append(m.best_iteration)
    tick(f"fold {fold}: best_iter={m.best_iteration}, val logloss={m.best_score['valid_0']['binary_logloss']:.4f}")
feat_tr["p_raw"] = oof

# ==== Cell 11: Calibration, one-home constraint, decision rule tuned for macro F0.5 ====
calib = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof, y)


def postprocess(df, calibrator, cfg=CFG):
    """Calibrate probabilities and, if one_home, keep each S2/S3 record only on its most likely S1."""
    df = df[["i", "j", "p_raw"]].copy()
    df["p"] = calibrator.predict(df["p_raw"].values)
    if cfg["one_home"]:
        best = df.groupby("j")["p"].transform("max")
        df.loc[df["p"] < best - 1e-12, "p"] = 0.0
    return df


def to_pred(df_sel, s1, ot):
    """Selected (i, j) rows -> {s1_entity_id: [candidate ids]}."""
    s1_ids, ot_ids = s1["entity_id"].values, ot["entity_id"].values
    out = {}
    for a, b in zip(df_sel["i"].values, df_sel["j"].values):
        out.setdefault(s1_ids[a], []).append(ot_ids[b])
    return out


def decide_threshold(df, t):
    """Predict every candidate with calibrated probability >= t."""
    return df[df["p"] >= t]


def decide_expected_f(df, gamma=1.0, beta=0.5, n_sims=1000, min_p=0.02, max_k=10, seed=0):
    """Per S1, choose the top-k set (k may be 0) with the highest EXPECTED F-beta under the model's
    probabilities (Monte-Carlo over possible truths). gamma > 1 makes the rule more conservative."""
    rng = np.random.default_rng(seed)
    d = df[df["p"] >= min_p].sort_values(["i", "p"], ascending=[True, False])
    d = d[d.groupby("i").cumcount() < max_k]
    b2 = beta ** 2
    keep = []
    starts = np.flatnonzero(np.r_[True, d["i"].values[1:] != d["i"].values[:-1]])
    ends = np.r_[starts[1:], len(d)]
    P = d["p"].values ** gamma
    for s, e in zip(starts, ends):
        p = P[s:e]
        sims = rng.random((n_sims, len(p))) < p
        n_true = sims.sum(1)
        best_k, best_f = 0, (n_true == 0).mean()
        for k in range(1, len(p) + 1):
            tp = sims[:, :k].sum(1)
            fval = np.where(tp > 0, (1 + b2) * tp / ((1 + b2) * tp + b2 * (n_true - tp) + (k - tp)), 0.0).mean()
            if fval > best_f:
                best_k, best_f = k, fval
        keep.extend(range(s, s + best_k))
    return d.iloc[keep]


pp_tr = postprocess(feat_tr, calib)
results = []
for t in np.arange(0.20, 0.96, 0.025):
    results.append(("threshold", round(float(t), 3), macro_f(to_pred(decide_threshold(pp_tr, t), tr_s1, tr_ot), truth)))
for g in (1.0, 1.25, 1.5):
    results.append(("expected_f", g, macro_f(to_pred(decide_expected_f(pp_tr, gamma=g), tr_s1, tr_ot), truth)))
res = pd.DataFrame(results, columns=["method", "param", "oof_macro_f05"]).sort_values("oof_macro_f05", ascending=False)
print(res.head(10).to_string(index=False))
BEST = dict(method=res.iloc[0]["method"], param=float(res.iloc[0]["param"]))
print("Chosen decision rule:", BEST)


def decide(df, best=None):
    """Apply the chosen decision rule."""
    best = best or BEST
    return decide_threshold(df, best["param"]) if best["method"] == "threshold" else decide_expected_f(df, gamma=best["param"])


pred_oof = to_pred(decide(pp_tr), tr_s1, tr_ot)
score_report(pred_oof, truth, label="OOF (all train S1):")
# Note: the calibrator is fitted on the same OOF predictions, so this score is slightly optimistic.

# ==== Cell 12 (optional): leave-one-country-out check — simulates an unseen country like France ====
ctry_of_pair = tr_s1["ckey"].values[feat_tr["i"].values]
n_rounds = int(np.mean(best_iters) * 1.1) + 1
for c in sorted(set(ctry_of_pair)):
    tr_m, va_m = ctry_of_pair != c, ctry_of_pair == c
    if tr_m.sum() == 0 or y[tr_m].sum() == 0:
        continue
    m = lgb.train(LGB_PARAMS, lgb.Dataset(X[tr_m], y[tr_m]), num_boost_round=n_rounds)
    df_c = feat_tr.loc[va_m, ["i", "j"]].copy()
    df_c["p_raw"] = m.predict(X[va_m])
    ids_c = tr_s1.loc[tr_s1["ckey"] == c, "entity_id"]
    score_report(to_pred(decide(postprocess(df_c, calib)), tr_s1, tr_ot), truth, ids=ids_c,
                 label=f"Train WITHOUT '{c}', test on '{c}':")
# A big drop versus Cell 11 means some feature/rule is country-specific; that will hurt on France.

# ==== Cell 13: Test set — normalize, block, featurize, predict ====
te_s1_raw, te_ot_raw = load_split("test")
te_s1 = add_norm_columns(te_s1_raw)
te_ot = add_norm_columns(te_ot_raw)
tx_te = SplitText(te_s1, te_ot)
pairs_te = generate_candidates(te_s1, te_ot, tx_te)
feat_te = build_features(pairs_te, te_s1, te_ot, tx_te)
tick(f"test: S1={len(te_s1):,}  S2+S3={len(te_ot):,}  candidate pairs={len(pairs_te):,}")
per_c = te_s1.loc[pairs_te["i"].unique(), "ckey"].value_counts() / te_s1["ckey"].value_counts()
print("Share of test S1 with >=1 candidate, by country:", per_c.round(3).to_dict())

feat_te["p_raw"] = np.mean([m.predict(feat_te[FEATS], num_iteration=m.best_iteration) for m in models], axis=0)
pp_te = postprocess(feat_te, calib)
sel_te = decide(pp_te)
pred_te = to_pred(sel_te, te_s1, te_ot)
cand_te = to_pred(pairs_te, te_s1, te_ot)      # the exact set the model scored = candidate_pairs.tsv
print(f"Test S1 with >=1 predicted match: {len(pred_te):,} / {len(te_s1):,}; "
      f"predicted pairs: {sum(map(len, pred_te.values())):,}")

# ==== Cell 14: Write output files and run the official validator ====
def write_id_lists(path, s1_ids, mapping, col):
    """One row per S1 id; comma-separated unique ids; empty string when none."""
    rows = [(sid, ",".join(sorted(set(mapping.get(sid, []))))) for sid in s1_ids]
    pd.DataFrame(rows, columns=["source1_entity_id", col]).to_csv(path, sep="\t", index=False, lineterminator="\n")


test_s1_ids = te_s1["entity_id"].tolist()
valid_other = set(te_ot["entity_id"])
assert len(test_s1_ids) == len(set(test_s1_ids)), "duplicate S1 ids in test_source1"
for sid, ms in pred_te.items():
    assert set(ms) <= set(cand_te.get(sid, [])), f"{sid}: match not in candidates"
    assert all(m in valid_other and m[:3] in ("S2-", "S3-") for m in ms), f"{sid}: invalid id"

MATCH_PATH, CAND_PATH = OUT_DIR / "matching_results.tsv", OUT_DIR / "candidate_pairs.tsv"
write_id_lists(MATCH_PATH, test_s1_ids, pred_te, "matched_entity_ids")
write_id_lists(CAND_PATH, test_s1_ids, cand_te, "candidate_entity_ids")
tick(f"wrote {MATCH_PATH} and {CAND_PATH}")

if VALIDATOR.exists():
    r = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(MATCH_PATH),
                        "--candidate", str(CAND_PATH), "--test-dir", str(TEST_DIR)],
                       cwd=str(BASE), capture_output=True, text=True)
    print(r.stdout, r.stderr)
else:
    print(f"Validator not found at {VALIDATOR}; copy utils/validate_submission.py there and re-run this cell.")

# ==== Cell 15: Save models, settings and feature importance ====
for k, m in enumerate(models):
    m.save_model(str(ART_DIR / f"lgb_fold{k}.txt"))
with open(ART_DIR / "calibrator.pkl", "wb") as fh:
    pickle.dump(calib, fh)
with open(ART_DIR / "run_config.json", "w") as fh:
    json.dump({"CFG": CFG, "BEST": BEST, "FEATS": FEATS, "best_iters": best_iters,
               "oof_macro_f05": macro_f(pred_oof, truth)}, fh, indent=2)
imp = pd.Series(np.mean([m.feature_importance("gain") for m in models], axis=0), index=FEATS)
print("Top 20 features by gain:\n", (imp / imp.sum()).sort_values(ascending=False).head(20).round(4).to_string())
tick("done")
