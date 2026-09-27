"""Business Entity Resolution - FAST pipeline (same code as er_fast_pipeline.ipynb).
Run from anywhere:  python run_pipeline.py   (or set ER_BASE=/path/to/student_resource)
"""
# ==== Cell 2: Imports, paths and configuration ====
import os, re, gc, sys, csv, json, time, pickle, random, subprocess, warnings
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


def find_base():
    """Locate student_resource/: ER_BASE env var first, then common places (Windows, Linux, macOS, SageMaker)."""
    if os.environ.get("ER_BASE"):
        return Path(os.environ["ER_BASE"])
    candidates = [Path.cwd(), Path.cwd().parent, Path.cwd() / "student_resource",
                  Path.home() / "SageMaker" / "student_resource",   # SageMaker Notebook Instance
                  Path("/home/sagemaker-user/student_resource"),    # SageMaker Studio
                  Path.home() / "student_resource", Path.home() / "Desktop" / "student_resource",
                  Path.home() / "Downloads" / "student_resource"]
    for c in candidates:
        if (c / "dataset" / "train" / "train_source1.tsv").exists():
            return c.resolve()
    return Path.cwd()


BASE = find_base()
# To set it by hand, uncomment one line, e.g.:
# BASE = Path(r"C:\Users\you\student_resource")
# BASE = Path("/home/ec2-user/SageMaker/student_resource")

TRAIN_DIR, TEST_DIR = BASE / "dataset" / "train", BASE / "dataset" / "test"
OUT_DIR, ART_DIR = BASE / "output", BASE / "artifacts"
VALIDATOR = BASE / "utils" / "validate_submission.py"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED = ([TRAIN_DIR / f"train_source{k}.tsv" for k in (1, 2, 3)] + [TRAIN_DIR / "train_ground_truth.tsv"]
            + [TEST_DIR / f"test_source{k}.tsv" for k in (1, 2, 3)])
missing = [str(p) for p in EXPECTED if not p.exists()]
if missing:
    raise FileNotFoundError("Missing files:\n  " + "\n  ".join(missing) + f"\nBASE = {BASE}  (set it by hand above)")
print("All data files found under", BASE)

CFG = dict(
    seed=42,
    # ---- time / memory budget ----
    max_train_s1=100_000,   # train on a scaled-down copy with at most this many S1 entities (None = use all)
    # ---- blocking ----
    k_name=15, k_addr=8, k_full=15, k_rev=3,   # top-K candidates per method
    max_df_name=500,        # blocking ignores name n-grams shared by more than this many records
    max_df_addr=500,        # same for addresses. Higher = better recall but slower
    max_cand_per_s1=30,     # keep at most this many candidates per S1 (best by TF-IDF cosine)
    postal_max_group=30,    # postal+token blocking skips keys shared by more records than this
    block_by_country=None,  # None = decided automatically in Cell 3
    one_home=None,          # None = decided automatically in Cell 3
    block_max_nnz=5e6,      # scipy fallback only: max non-zeros per chunk
    # ---- model ----
    n_folds=3, max_rounds=1000, learning_rate=0.08,
    feature_chunk=1_000_000,  # test features are built and scored in chunks of this many pairs
    beta=0.5,
)
random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
T0 = time.time()


def peak_ram_mb():
    """Peak memory of this Python process in MB (works on Windows, Linux and macOS)."""
    try:
        import resource                          # Linux / macOS
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1024 / 1024 if sys.platform == "darwin" else r / 1024
    except ImportError:
        pass
    try:
        import psutil                            # Windows
        mi = psutil.Process().memory_info()
        return (getattr(mi, "peak_wset", None) or mi.rss) / 1024 / 1024
    except ImportError:
        return float("nan")


def tick(msg):
    """Print a message with elapsed time and peak RAM."""
    print(f"[{time.time() - T0:7.1f}s | peak RAM {peak_ram_mb():,.0f} MB] {msg}", flush=True)


tick(f"CPU cores: {os.cpu_count()}")

# ==== Cell 3: Load data, automatic checks, scaled-down training copy ====
def read_tsv(path):
    """Read a challenge TSV with every column as a string (QUOTE_NONE: stray quotes can't merge lines)."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    df.columns = [c.strip().strip('"') for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip().str.strip('"')
    return df


def load_split(split):
    """Return (source1, others) for 'train' or 'test'; others = Source 2 + Source 3 with a 'source' column."""
    d = TRAIN_DIR if split == "train" else TEST_DIR
    s1 = read_tsv(d / f"{split}_source1.tsv")
    ot = pd.concat([read_tsv(d / f"{split}_source2.tsv"), read_tsv(d / f"{split}_source3.tsv")], ignore_index=True)
    for df in (s1, ot):
        for col in ("business_name", "business_address", "country"):
            if col not in df.columns:
                df[col] = ""
        df["ckey"] = df["country"].str.strip().str.lower().replace("", "unknown")   # open set of countries
    ot["source"] = ot["entity_id"].str[:2]
    return s1.reset_index(drop=True), ot.reset_index(drop=True)


def load_truth(s1_ids):
    """Ground truth as {s1_id: set(matched ids)}; every S1 id present (empty set = singleton)."""
    gt = read_tsv(TRAIN_DIR / "train_ground_truth.tsv")
    truth = {sid: set() for sid in s1_ids}
    for sid, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        truth.setdefault(sid, set()).update(x.strip() for x in m.split(",") if x.strip())
    return truth


tr_s1, tr_ot = load_split("train")
truth = load_truth(tr_s1["entity_id"])
tick(f"train loaded: S1={len(tr_s1):,}  S2+S3={len(tr_ot):,}")

# --- automatic checks (these set CFG values left as None) ---
n_match = np.array([len(v) for v in truth.values()])
all_matched = [m for v in truth.values() for m in v]
multi = len(all_matched) - len(set(all_matched))
c1, c2 = dict(zip(tr_s1.entity_id, tr_s1.ckey)), dict(zip(tr_ot.entity_id, tr_ot.ckey))
pairs_list = [(s, m) for s, v in truth.items() for m in v]
mismatch = sum(c1.get(s) != c2.get(m) for s, m in pairs_list) / max(1, len(pairs_list))
if CFG["one_home"] is None:
    CFG["one_home"] = multi == 0
if CFG["block_by_country"] is None:
    CFG["block_by_country"] = mismatch < 0.01
print(f"singleton rate {np.mean(n_match == 0):.1%} | mean matches {n_match.mean():.2f} | "
      f"S2/S3 ids under >1 S1: {multi} -> one_home={CFG['one_home']} | "
      f"country mismatch in true pairs {mismatch:.2%} -> block_by_country={CFG['block_by_country']}")

# --- scaled-down copy of train: sample S1, keep their matches, keep the same share of unmatched S2/S3 ---
if CFG["max_train_s1"] and len(tr_s1) > CFG["max_train_s1"]:
    frac = CFG["max_train_s1"] / len(tr_s1)
    keep_s1 = tr_s1.sample(n=CFG["max_train_s1"], random_state=CFG["seed"])
    keep_matched = {m for s in keep_s1["entity_id"] for m in truth[s]}
    unmatched = tr_ot[~tr_ot["entity_id"].isin(set(all_matched))]
    keep_rows = tr_ot["entity_id"].isin(keep_matched) | tr_ot.index.isin(
        unmatched.sample(frac=frac, random_state=CFG["seed"]).index)
    tr_s1 = keep_s1.sort_index().reset_index(drop=True)
    tr_ot = tr_ot[keep_rows].reset_index(drop=True)
    truth = {s: truth[s] for s in tr_s1["entity_id"]}
    tick(f"using scaled-down train copy: S1={len(tr_s1):,}  S2+S3={len(tr_ot):,}")
true_pairs = [(s, m) for s, v in truth.items() for m in v]
del c1, c2, pairs_list, all_matched

# ==== Cell 4: Text normalization (hand-written rules only; results cached per unique string) ====
LEGAL = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "cie": "co", "compagnie": "co",
    "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt", "pte": "pvt",
    "llc": "llc", "llp": "llp", "plc": "plc", "lp": "lp", "pllc": "pllc", "opc": "opc", "gmbh": "gmbh",
    "sa": "sa", "sas": "sas", "sasu": "sasu", "sarl": "sarl", "eurl": "eurl", "snc": "snc", "sci": "sci",
    "selarl": "selarl", "scop": "scop",
}
NAME_CANON = {
    "technologies": "tech", "technology": "tech", "techno": "tech",
    "enterprises": "ent", "enterprise": "ent", "services": "svc", "service": "svc", "svcs": "svc",
    "international": "intl", "national": "natl", "manufacturing": "mfg", "management": "mgmt",
    "associates": "assoc", "brothers": "bros", "industries": "inds", "group": "grp", "holdings": "hldgs",
    "solutions": "sol", "systems": "sys", "pharmaceuticals": "pharma", "pharmaceutical": "pharma",
    "center": "ctr", "centre": "ctr", "etablissements": "ets", "university": "univ", "hospital": "hosp",
    "saint": "st",
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
MULTI_WORD = {
    "new york": "ny", "new jersey": "nj", "new mexico": "nm", "new hampshire": "nh", "north carolina": "nc",
    "north dakota": "nd", "south carolina": "sc", "south dakota": "sd", "west virginia": "wv",
    "rhode island": "ri", "district of columbia": "dc", "new delhi": "delhi",
}
CITY_ALIAS = {
    "bangalore": "bengaluru", "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai",
    "gurgaon": "gurugram", "poona": "pune", "baroda": "vadodara", "trivandrum": "thiruvananthapuram",
    "mysore": "mysuru", "mangalore": "mangaluru", "vizag": "visakhapatnam", "cochin": "kochi",
    "pondicherry": "puducherry",
}
LANDMARK = {"nr", "opp", "bhd", "beside", "next", "adjacent", "landmark", "pres", "face"}
_MULTI_RE = re.compile(r"\b(" + "|".join(map(re.escape, MULTI_WORD)) + r")\b")
_PIN_SPLIT = re.compile(r"\b(\d{3}) (\d{3})\b")
_POSTAL = re.compile(r"^\d{5,6}$")


def base_clean(s):
    """Lowercase, strip accents, unify punctuation: 'P.V.T.' -> 'pvt', '&' -> 'and'."""
    s = unidecode(str(s)).lower()
    s = re.sub(r"(?<=[a-z])\.(?=[a-z])", "", s)
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
    s = _PIN_SPLIT.sub(r"\1\2", base_clean(raw))
    s = _MULTI_RE.sub(lambda m: MULTI_WORD[m.group(1)], s)
    toks = []
    for t in s.split():
        t = CITY_ALIAS.get(t, t)          # bombay -> mumbai
        t = US_STATES.get(t, t)           # california -> ca
        toks.append(ADDR_CANON.get(t, t)) # road -> rd, street -> st

    pos = [k for k, t in enumerate(toks) if _POSTAL.match(t)]
    postal = toks[pos[-1]] if pos and not (pos[-1] == 0 and len(toks) > 1) else ""
    nums = [t for t in toks if any(ch.isdigit() for ch in t) and t != postal]
    landmark = int(any(t in LANDMARK for t in toks))
    return " ".join(toks), postal, " ".join(sorted(set(nums))), (nums[0] if nums else ""), landmark


def name_extras(core):
    """(compact, acronym, first token, phonetic key) for a normalized core name."""
    toks = core.split()
    acr = "".join(t[0] for t in toks) if len(toks) >= 2 else ""
    phon = " ".join(jellyfish.metaphone(t) for t in toks if t.isalpha())
    return core.replace(" ", ""), acr, (toks[0] if toks else ""), phon


def cached_map(values, fn):
    """Apply fn once per unique value (big speed-up when strings repeat)."""
    cache = {u: fn(u) for u in pd.unique(np.asarray(values, dtype=object))}
    return [cache[v] for v in values]


def add_norm_columns(df):
    """Add all normalized columns used by blocking and features."""
    df = df.copy()
    nm = cached_map(df["business_name"].values, normalize_name)
    df["name_core"] = [a for a, _ in nm]
    df["legal"] = [b for _, b in nm]
    df["name_clean"] = cached_map(df["business_name"].values, base_clean)
    ex = cached_map(df["name_core"].values, name_extras)
    df["name_compact"], df["name_acr"], df["name_first"], df["name_phon"] = [list(x) for x in zip(*ex)] if len(ex) else ([], [], [], [])
    ad = cached_map(df["business_address"].values, normalize_address)
    df["addr_norm"], df["postal"], df["nums"], df["first_num"], df["landmark"] = [list(x) for x in zip(*ad)] if len(ad) else ([],) * 5
    df["landmark"] = df["landmark"].astype(np.int8)
    return df


tr_s1, tr_ot = add_norm_columns(tr_s1), add_norm_columns(tr_ot)
tick("normalized train")
print(tr_ot[["business_name", "name_core", "legal", "business_address", "addr_norm", "postal"]].head(5).to_string())

# ==== Cell 5: Blocking functions (TF-IDF top-K with capped n-gram frequency, postal+token keys) ====
try:
    from sparse_dot_topn import sp_matmul_topn   # optional, MIT licence, multithreaded C++
    HAVE_TOPN = True
except ImportError:
    HAVE_TOPN = False
print("sparse_dot_topn available:", HAVE_TOPN)


def _l2_rows(M):
    """L2-normalize rows of a sparse matrix (all-zero rows stay zero)."""
    n = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
    n[n == 0] = 1.0
    return sp.diags(1.0 / n).dot(M).astype(np.float32).tocsr()


class SplitText:
    """TF-IDF matrices for one split, fitted on that split's own text (no labels).
    Full vectors N/A/W feed the features. Blocking vectors bN/bA/bF drop any n-gram shared by more than
    max_df_* records: each record can then only 'collide' with a bounded number of others, which keeps
    blocking time roughly linear in the data size."""

    def __init__(self, s1, ot, cfg=CFG):
        def fit(col, **kw):
            text = pd.concat([s1[col], ot[col]], ignore_index=True)
            try:
                v = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, **kw).fit(text)
            except ValueError:            # tiny data where pruning removes everything
                kw.pop("max_df", None)
                v = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, **kw).fit(text)
            return v.transform(s1[col]).tocsr(), v.transform(ot[col]).tocsr()

        char = dict(analyzer="char_wb", ngram_range=(3, 4))
        self.N1, self.N2 = fit("name_core", **char)
        self.A1, self.A2 = fit("addr_norm", **char)
        self.W1, self.W2 = fit("name_core", analyzer="word", token_pattern=r"(?u)\b\w+\b")
        self.bN1, self.bN2 = fit("name_core", max_df=int(cfg["max_df_name"]), **char)
        self.bA1, self.bA2 = fit("addr_norm", max_df=int(cfg["max_df_addr"]), **char)
        self.bF1 = _l2_rows(sp.hstack([self.bN1, self.bA1], format="csr"))
        self.bF2 = _l2_rows(sp.hstack([self.bN2, self.bA2], format="csr"))


def topk_sparse(Q, D, k, cfg=CFG, label=""):
    """Top-k rows of D for every row of Q by dot product -> (q_idx, d_idx, score). Never builds a dense matrix."""
    empty = (np.array([], np.int64), np.array([], np.int64), np.array([], np.float32))
    nq, nd = Q.shape[0], D.shape[0]
    if nq == 0 or nd == 0:
        return empty
    k, DT, t0 = min(k, nd), D.T.tocsr(), time.time()
    if HAVE_TOPN:
        C = sp_matmul_topn(Q, DT, top_n=k, sort=False, n_threads=os.cpu_count()).tocoo()
        keep = C.data > 0
        out = (C.row[keep].astype(np.int64), C.col[keep].astype(np.int64), C.data[keep])
    else:
        rows, cols, vals = [], [], []
        st, step = 0, min(nq, 500)
        while st < nq:
            S = (Q[st:st + step] @ DT).tocsr()
            if S.nnz:
                r = np.repeat(np.arange(S.shape[0]), np.diff(S.indptr))
                order = np.lexsort((-S.data, r))
                r_sorted = r[order]
                rank = np.arange(len(order)) - S.indptr[r_sorted]
                keep = (rank < k) & (S.data[order] > 0)
                rows.append(st + r_sorted[keep])
                cols.append(S.indices[order][keep].astype(np.int64))
                vals.append(S.data[order][keep])
            per_row = max(1.0, S.nnz / max(1, S.shape[0]))
            st += S.shape[0]
            step = int(np.clip(cfg["block_max_nnz"] / per_row, 50, 50_000))
            del S
        out = empty if not rows else (np.concatenate(rows), np.concatenate(cols), np.concatenate(vals))
    if label:
        print(f"    {label:8s} {nq:>9,} x {nd:<9,} -> {len(out[0]):>10,} pairs ({time.time() - t0:.1f}s)", flush=True)
    return out


def postal_token_pairs(s1, ot, i_idx, j_idx, max_group):
    """Pairs sharing the same postal code AND an uncommon name token."""
    def long_table(df, idx):
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
        return np.array([], np.int64), np.array([], np.int64)
    df_tok = pd.concat([L["tok"], R["tok"]]).value_counts()
    common = set(df_tok[df_tok > max(20, 0.01 * (len(i_idx) + len(j_idx)))].index)
    L = L[~L["tok"].isin(common)].assign(key=lambda d: d["postal"] + "|" + d["tok"])
    R = R[~R["tok"].isin(common)].assign(key=lambda d: d["postal"] + "|" + d["tok"])
    cntL, cntR = L["key"].value_counts(), R["key"].value_counts()
    ok = set(cntR[cntR <= max_group].index) & set(cntL[cntL <= max_group].index)
    m = L[L["key"].isin(ok)][["row", "key"]].merge(R[R["key"].isin(ok)][["row", "key"]], on="key",
                                                    suffixes=("_i", "_j")).drop_duplicates(["row_i", "row_j"])
    return m["row_i"].values.astype(np.int64), m["row_j"].values.astype(np.int64)


METHODS = ["m_name", "m_addr", "m_full", "m_rev", "m_postal"]


def rowwise_dot(X, Y, i, j, chunk=500_000):
    """Cosine of X[i] with Y[j] for every pair (rows are L2-normalized)."""
    out = np.empty(len(i), dtype=np.float32)
    for s in range(0, len(i), chunk):
        out[s:s + chunk] = np.asarray(X[i[s:s + chunk]].multiply(Y[j[s:s + chunk]]).sum(axis=1)).ravel()
    return out


def generate_candidates(s1, ot, tx, cfg=CFG):
    """Union of 5 blocking methods inside each country label, then keep the best max_cand_per_s1 per S1.
    Returns DataFrame (i, j, m_*, n_char_cos, a_char_cos, full_cos) with i/j = row positions in s1/ot."""
    n_ot = len(ot)
    keys, bits = [], []

    def add(bit, gi, gj):
        keys.append(np.asarray(gi, np.int64) * n_ot + np.asarray(gj, np.int64))
        bits.append(np.full(len(gi), 1 << bit, np.uint8))

    for c in sorted(s1["ckey"].unique()):
        i_idx = np.where(s1["ckey"].values == c)[0]
        j_idx = np.where(ot["ckey"].values == c)[0] if cfg["block_by_country"] else np.arange(n_ot)
        if len(i_idx) == 0 or len(j_idx) == 0:
            continue
        tick(f"blocking '{c}': {len(i_idx):,} S1 x {len(j_idx):,} S2/S3")
        for bit, (name, M1, M2, k) in enumerate((("name", tx.bN1, tx.bN2, cfg["k_name"]),
                                                  ("address", tx.bA1, tx.bA2, cfg["k_addr"]),
                                                  ("full", tx.bF1, tx.bF2, cfg["k_full"]))):
            q, d, _ = topk_sparse(M1[i_idx], M2[j_idx], k, label=name)
            add(bit, i_idx[q], j_idx[d])
        q, d, _ = topk_sparse(tx.bN2[j_idx], tx.bN1[i_idx], cfg["k_rev"], label="reverse")
        add(3, i_idx[d], j_idx[q])
        pi, pj = postal_token_pairs(s1, ot, i_idx, j_idx, cfg["postal_max_group"])
        print(f"    postal   -> {len(pi):>10,} pairs", flush=True)
        add(4, pi, pj)
        gc.collect()

    cols = ["i", "j"] + METHODS + ["n_char_cos", "a_char_cos", "full_cos"]
    if not keys:
        return pd.DataFrame({c: pd.Series(dtype=np.float32) for c in cols}).astype({"i": np.int32, "j": np.int32})
    keys, bits = np.concatenate(keys), np.concatenate(bits)
    order = np.argsort(keys, kind="stable")
    keys, bits = keys[order], bits[order]
    uniq, first = np.unique(keys, return_index=True)
    mask = np.bitwise_or.reduceat(bits, first)
    del keys, bits, order
    pairs = pd.DataFrame({"i": (uniq // n_ot).astype(np.int32), "j": (uniq % n_ot).astype(np.int32)})
    for b, m in enumerate(METHODS):
        pairs[m] = ((mask >> b) & 1).astype(np.int8)
    # full (unpruned) cosines, then keep the best max_cand_per_s1 candidates per S1
    i, j = pairs["i"].values, pairs["j"].values
    pairs["n_char_cos"] = rowwise_dot(tx.N1, tx.N2, i, j)
    pairs["a_char_cos"] = rowwise_dot(tx.A1, tx.A2, i, j)
    pairs["full_cos"] = (pairs["n_char_cos"] + pairs["a_char_cos"]) / 2
    score = np.maximum(pairs["full_cos"].values, np.maximum(pairs["n_char_cos"].values, pairs["a_char_cos"].values))
    pairs = pairs.assign(_s=score).sort_values(["i", "_s"], ascending=[True, False])
    pairs = pairs[pairs.groupby("i").cumcount() < cfg["max_cand_per_s1"]].drop(columns="_s")
    return pairs.sort_values(["i", "j"]).reset_index(drop=True)

# ==== Cell 6: Run blocking on train and measure recall ====
tx_tr = SplitText(tr_s1, tr_ot)
tick("train TF-IDF built")
pairs_tr = generate_candidates(tr_s1, tr_ot, tx_tr)
tick(f"train candidate pairs: {len(pairs_tr):,}")

s1_pos = pd.Series(np.arange(len(tr_s1)), index=tr_s1["entity_id"])
ot_pos = pd.Series(np.arange(len(tr_ot)), index=tr_ot["entity_id"])
tp_df = pd.DataFrame(true_pairs, columns=["s1", "ot"])
tp_df = tp_df[tp_df["s1"].isin(s1_pos.index) & tp_df["ot"].isin(ot_pos.index)]
true_i = s1_pos.loc[tp_df["s1"]].values.astype(np.int64)
true_j = ot_pos.loc[tp_df["ot"]].values.astype(np.int64)
pairs_tr["label"] = np.isin(pairs_tr["i"].values.astype(np.int64) * len(tr_ot) + pairs_tr["j"].values,
                            true_i * len(tr_ot) + true_j).astype(np.int8)

n_true, found = len(true_i), int(pairs_tr["label"].sum())
print(f"Pair recall (recall ceiling): {found / max(1, n_true):.4f}  ({found:,} of {n_true:,} true pairs)")
for m in METHODS:
    print(f"  {m:9s} alone: recall {pairs_tr.loc[pairs_tr[m] == 1, 'label'].sum() / max(1, n_true):.4f}")
tp = pairs_tr[pairs_tr["label"] == 1]
print("  recall by country:", (pd.Series(tr_s1["ckey"].values[tp["i"].values]).value_counts()
                               / pd.Series(tr_s1["ckey"].values[true_i]).value_counts()).round(4).to_dict())
per = np.bincount(pairs_tr["i"].values, minlength=len(tr_s1))
print(f"  candidates per S1: mean {per.mean():.1f}, max {per.max()}")
# Recall below ~0.95? Raise max_df_name / max_df_addr (e.g. 1000) or k_name/k_full, then re-run cells 5-6.

# ==== Cell 7: Pair features (cheap ones on all pairs, detailed ones per chunk) ====
def group_context(f, key, col, prefix):
    """Rank / gap-to-best / margin-over-best-other of `col` within each group of `key`."""
    g = f.groupby(key)[col]
    f[f"{prefix}_rank"] = g.rank(ascending=False, method="min").astype(np.float32)
    f[f"{prefix}_gap"] = (f[col] - g.transform("max")).astype(np.float32)
    s = f[[key, col]].sort_values([key, col], ascending=[True, False])
    s["r"] = s.groupby(key).cumcount()
    top1 = s[s["r"] == 0].set_index(key)[col]
    top2 = s[s["r"] == 1].set_index(key)[col]
    t1, t2 = f[key].map(top1).values, f[key].map(top2).fillna(-1).values
    f[f"{prefix}_margin"] = (f[col].values - np.where(f[col].values >= t1, t2, t1)).astype(np.float32)


def cheap_features(pairs, ot, tx):
    """Features that are fast for millions of pairs: TF-IDF cosines, blocking flags, ranking context."""
    f = pairs.copy()
    f["n_word_cos"] = rowwise_dot(tx.W1, tx.W2, f["i"].values, f["j"].values)
    f["n_methods"] = f[METHODS].sum(axis=1).astype(np.int8)
    f["is_s3"] = (ot["source"].values[f["j"].values] == "S3").astype(np.int8)
    for col in ("full_cos", "n_char_cos", "a_char_cos", "n_word_cos"):
        group_context(f, "i", col, f"gi_{col}")   # this candidate vs the S1's other candidates
    for col in ("full_cos", "n_char_cos"):
        group_context(f, "j", col, f"gj_{col}")   # this S1 vs the candidate's other possible S1s
    f["n_cand_i"] = f.groupby("i")["j"].transform("size").astype(np.int32)
    f["n_cand_j"] = f.groupby("j")["i"].transform("size").astype(np.int32)
    return f


def pair_sim(a, b, scorer, scale=100.0):
    """Element-wise string similarity of two equal-length lists, scaled to 0-1 (multithreaded)."""
    try:
        r = process.cpdist(list(a), list(b), scorer=scorer, workers=-1)
    except Exception:
        r = np.array([scorer(x, y) for x, y in zip(a, b)])
    return np.asarray(r, dtype=np.float32) / scale


DETAIL_COLS = ["name_core", "name_compact", "name_phon", "name_clean", "name_first", "name_acr", "legal",
               "addr_norm", "postal", "nums", "first_num", "landmark"]


def detail_features(pairs, s1, ot):
    """String-similarity features (rapidfuzz, postal/number logic, legal suffix). Index matches `pairs`."""
    i, j = pairs["i"].values, pairs["j"].values
    A = s1[DETAIL_COLS].iloc[i].reset_index(drop=True)
    B = ot[DETAIL_COLS].iloc[j].reset_index(drop=True)
    f = pd.DataFrame(index=range(len(pairs)))
    f["n_ratio"] = pair_sim(A.name_core, B.name_core, fuzz.ratio)
    f["n_tset"] = pair_sim(A.name_core, B.name_core, fuzz.token_set_ratio)
    f["n_tsort"] = pair_sim(A.name_core, B.name_core, fuzz.token_sort_ratio)
    f["n_partial"] = pair_sim(A.name_core, B.name_core, fuzz.partial_ratio)
    f["n_jw"] = pair_sim(A.name_core, B.name_core, JaroWinkler.normalized_similarity, scale=1.0)
    f["n_compact"] = pair_sim(A.name_compact, B.name_compact, fuzz.ratio)
    f["n_phon"] = pair_sim(A.name_phon, B.name_phon, fuzz.token_set_ratio)
    f["n_raw"] = pair_sim(A.name_clean, B.name_clean, fuzz.ratio)
    f["n_first_eq"] = (A.name_first.values == B.name_first.values).astype(np.int8)
    f["n_acronym"] = (((A.name_acr.values != "") & (A.name_acr.values == B.name_compact.values)) |
                      ((B.name_acr.values != "") & (B.name_acr.values == A.name_compact.values))).astype(np.int8)
    f["n_exact"] = (A.name_core.values == B.name_core.values).astype(np.int8)
    la, lb = A.name_core.str.len().values, B.name_core.str.len().values
    f["n_len_min"], f["n_len_diff"] = np.minimum(la, lb), np.abs(la - lb)
    lg_a, lg_b = A.legal.values, B.legal.values
    both = (lg_a != "") & (lg_b != "")
    f["legal_both"] = both.astype(np.int8)
    f["legal_eq"] = (both & (lg_a == lg_b)).astype(np.int8)
    f["legal_conflict"] = (both & (lg_a != lg_b)).astype(np.int8)
    f["a_tset"] = pair_sim(A.addr_norm, B.addr_norm, fuzz.token_set_ratio)
    f["a_partial"] = pair_sim(A.addr_norm, B.addr_norm, fuzz.partial_ratio)
    f["a_ratio"] = pair_sim(A.addr_norm, B.addr_norm, fuzz.ratio)
    f["a_empty_any"] = ((A.addr_norm.values == "") | (B.addr_norm.values == "")).astype(np.int8)
    pa, pb = A.postal.values, B.postal.values
    pboth = (pa != "") & (pb != "")
    f["postal_both"] = pboth.astype(np.int8)
    f["postal_eq"] = (pboth & (pa == pb)).astype(np.int8)
    f["postal_conflict"] = (pboth & (pa != pb)).astype(np.int8)
    f["postal_pref3_eq"] = (pboth & (A.postal.str[:3].values == B.postal.str[:3].values)).astype(np.int8)
    f["postal_one_missing"] = ((pa != "") ^ (pb != "")).astype(np.int8)
    f["num_sim"] = pair_sim(A.nums, B.nums, fuzz.token_set_ratio)
    fa, fb = A.first_num.values, B.first_num.values
    f["first_num_eq"] = ((fa != "") & (fa == fb)).astype(np.int8)
    f["first_num_conflict"] = ((fa != "") & (fb != "") & (fa != fb)).astype(np.int8)
    f["landmark_any"] = (A.landmark.values | B.landmark.values).astype(np.int8)
    return f


feat_tr = cheap_features(pairs_tr, tr_ot, tx_tr)
feat_tr = pd.concat([feat_tr.reset_index(drop=True), detail_features(feat_tr, tr_s1, tr_ot)], axis=1)
FEATS = [c for c in feat_tr.columns if c not in ("i", "j", "label")]
tick(f"train features: {len(feat_tr):,} pairs x {len(FEATS)} features, positives {feat_tr['label'].sum():,}")

# ==== Cell 8: Competition metric (macro F0.5 per S1 entity) ====
def f_beta_entity(pred, true, beta=0.5):
    """F-beta for one S1 entity. Singleton: 1.0 if pred empty else 0.0."""
    if not true:
        return 1.0 if not pred else 0.0
    tp_ = len(pred & true)
    if tp_ == 0:
        return 0.0
    p, r, b2 = tp_ / len(pred), tp_ / len(true), beta ** 2
    return (1 + b2) * p * r / (b2 * p + r)


def macro_f_vec(sel, i, label, n_true_per_i, beta=0.5):
    """Vectorized macro F-beta over ALL S1 rows. sel/i/label align with candidate rows;
    n_true_per_i counts every true match of each S1 (also ones blocking missed)."""
    n = len(n_true_per_i)
    tp_ = np.bincount(i[sel], weights=label[sel], minlength=n)
    npred = np.bincount(i[sel], minlength=n)
    f = np.zeros(n)
    single = n_true_per_i == 0
    f[single] = npred[single] == 0
    m = ~single & (tp_ > 0)
    p, r, b2 = tp_[m] / npred[m], tp_[m] / n_true_per_i[m], beta ** 2
    f[m] = (1 + b2) * p * r / (b2 * p + r)
    return f.mean(), f


assert abs(f_beta_entity({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"}) - 0.714) < 1e-3
N_TRUE_TR = np.array([len(truth[s]) for s in tr_s1["entity_id"]])
# cross-check the vectorized metric against the reference one on random predictions
_rng = np.random.default_rng(0)
_sel = _rng.random(len(feat_tr)) < 0.1
_vec, _ = macro_f_vec(_sel, feat_tr["i"].values, feat_tr["label"].values, N_TRUE_TR)
_ids, _ot_ids = tr_s1["entity_id"].values, tr_ot["entity_id"].values
_pred = {}
for a, b in zip(feat_tr["i"].values[_sel], feat_tr["j"].values[_sel]):
    _pred.setdefault(_ids[a], set()).add(_ot_ids[b])
_ref = np.mean([f_beta_entity(_pred.get(s, set()), truth[s]) for s in _ids])
assert abs(_vec - _ref) < 1e-9, (_vec, _ref)
print(f"metric OK | 'predict nothing' baseline = {np.mean(N_TRUE_TR == 0):.4f}")

# ==== Cell 9: LightGBM, GroupKFold by S1 entity ====
LGB_PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=CFG["learning_rate"],
                  num_leaves=63, min_child_samples=40, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, max_bin=127, force_col_wise=True,
                  seed=CFG["seed"], verbose=-1, n_jobs=-1)
X = feat_tr[FEATS].to_numpy(dtype=np.float32)
y = feat_tr["label"].to_numpy()
groups = feat_tr["i"].to_numpy()
tick(f"training matrix {X.shape} ({X.nbytes / 1e6:,.0f} MB)")

oof = np.zeros(len(y))
models, best_iters = [], []
for fold, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=CFG["n_folds"]).split(X, y, groups)):
    dtr = lgb.Dataset(X[tr_idx], y[tr_idx], feature_name=FEATS, free_raw_data=True)
    dva = lgb.Dataset(X[va_idx], y[va_idx], reference=dtr, free_raw_data=True)
    m = lgb.train(LGB_PARAMS, dtr, num_boost_round=CFG["max_rounds"], valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
    oof[va_idx] = m.predict(X[va_idx], num_iteration=m.best_iteration)
    models.append(m)
    best_iters.append(m.best_iteration)
    del dtr, dva
    gc.collect()
    tick(f"fold {fold}: best_iter={m.best_iteration}")

# ==== Cell 10: Calibration, one-home rule, decision rule tuned for macro F0.5 ====
calib = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof, y)


def adjusted_probs(i, j, p_raw, cfg=CFG):
    """Calibrated probability; with one_home each S2/S3 record keeps probability only on its best S1.
    Also returns the rank of each candidate within its S1 (1 = most likely)."""
    d = pd.DataFrame({"i": i, "j": j, "p": calib.predict(p_raw)})
    if cfg["one_home"]:
        d.loc[d["p"] < d.groupby("j")["p"].transform("max") - 1e-12, "p"] = 0.0
    rank = d.groupby("i")["p"].rank(ascending=False, method="first").to_numpy()
    return d["p"].to_numpy(), rank


def select(p, rank, rule):
    """Predict a candidate when p >= threshold and it is among the S1's top max_k."""
    return (p >= rule["t"]) & (rank <= rule["max_k"])


p_tr, rank_tr = adjusted_probs(feat_tr["i"].values, feat_tr["j"].values, oof)
lab_tr, i_tr = feat_tr["label"].values, feat_tr["i"].values
grid = []
for t in np.arange(0.10, 0.96, 0.025):
    for mk in (1, 2, 3, 5, 1000):
        rule = {"t": round(float(t), 3), "max_k": mk}
        grid.append((macro_f_vec(select(p_tr, rank_tr, rule), i_tr, lab_tr, N_TRUE_TR)[0], rule))
grid.sort(key=lambda x: -x[0])
BEST = grid[0][1]
print("top rules:", [(round(s, 4), r) for s, r in grid[:5]])
score, f_each = macro_f_vec(select(p_tr, rank_tr, BEST), i_tr, lab_tr, N_TRUE_TR)
print(f"OOF macro F0.5 = {score:.4f} | singletons {f_each[N_TRUE_TR == 0].mean():.4f} "
      f"| non-singletons {f_each[N_TRUE_TR > 0].mean():.4f}   (rule {BEST})")
# The calibrator is fitted on these same OOF predictions, so this score is slightly optimistic.

# ==== Cell 11: Test set — normalize, block, features (in chunks), predict ====
te_s1, te_ot = load_split("test")
te_s1, te_ot = add_norm_columns(te_s1), add_norm_columns(te_ot)
tick(f"test loaded + normalized: S1={len(te_s1):,}  S2+S3={len(te_ot):,}")
tx_te = SplitText(te_s1, te_ot)
pairs_te = generate_candidates(te_s1, te_ot, tx_te)
tick(f"test candidate pairs: {len(pairs_te):,}")
has_cand = pd.Series(np.bincount(pairs_te["i"].values, minlength=len(te_s1)) > 0)
print("share of test S1 with >=1 candidate, by country:",
      has_cand.groupby(te_s1["ckey"].values).mean().round(3).to_dict())

feat_te = cheap_features(pairs_te, te_ot, tx_te).reset_index(drop=True)
p_raw_te = np.zeros(len(feat_te))
for st in range(0, len(feat_te), CFG["feature_chunk"]):
    part = feat_te.iloc[st:st + CFG["feature_chunk"]].reset_index(drop=True)
    Xc = pd.concat([part, detail_features(part, te_s1, te_ot)], axis=1)[FEATS].to_numpy(dtype=np.float32)
    p_raw_te[st:st + len(part)] = np.mean([m.predict(Xc, num_iteration=m.best_iteration) for m in models], axis=0)
    del part, Xc
    gc.collect()
    tick(f"scored test pairs {st + 1:,}-{min(st + CFG['feature_chunk'], len(feat_te)):,}")

p_te, rank_te = adjusted_probs(feat_te["i"].values, feat_te["j"].values, p_raw_te)
sel_te = select(p_te, rank_te, BEST)
te_ids, te_ot_ids = te_s1["entity_id"].values, te_ot["entity_id"].values
pred_te, cand_te = {}, {}
for a, b, s in zip(feat_te["i"].values, feat_te["j"].values, sel_te):
    cand_te.setdefault(te_ids[a], []).append(te_ot_ids[b])     # everything the model scored
    if s:
        pred_te.setdefault(te_ids[a], []).append(te_ot_ids[b])
print(f"test S1 with >=1 predicted match: {len(pred_te):,} / {len(te_s1):,}; predicted pairs {int(sel_te.sum()):,}")

# ==== Cell 12: Write output files and run the official validator ====
def write_id_lists(path, s1_ids, mapping, col):
    """One row per S1 id; comma-separated unique ids; empty when none."""
    rows = [(sid, ",".join(sorted(set(mapping.get(sid, []))))) for sid in s1_ids]
    pd.DataFrame(rows, columns=["source1_entity_id", col]).to_csv(path, sep="\t", index=False, lineterminator="\n")


ids = te_s1["entity_id"].tolist()
valid_other = set(te_ot["entity_id"])
assert len(ids) == len(set(ids)), "duplicate S1 ids in test_source1"
for sid, ms in pred_te.items():
    assert set(ms) <= set(cand_te[sid]), f"{sid}: match not among candidates"
    assert all(x in valid_other and x[:3] in ("S2-", "S3-") for x in ms), f"{sid}: invalid id"

MATCH_PATH, CAND_PATH = OUT_DIR / "matching_results.tsv", OUT_DIR / "candidate_pairs.tsv"
write_id_lists(MATCH_PATH, ids, pred_te, "matched_entity_ids")
write_id_lists(CAND_PATH, ids, cand_te, "candidate_entity_ids")
tick(f"wrote {MATCH_PATH} and {CAND_PATH}")

if VALIDATOR.exists():
    r = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(MATCH_PATH), "--candidate", str(CAND_PATH),
                        "--test-dir", str(TEST_DIR)], cwd=str(BASE), capture_output=True, text=True)
    print(r.stdout, r.stderr)
else:
    print(f"Validator not found at {VALIDATOR}; copy utils/validate_submission.py there and re-run this cell.")

# ==== Cell 13: Save models, settings and feature importance ====
for k, m in enumerate(models):
    m.save_model(str(ART_DIR / f"lgb_fold{k}.txt"))
with open(ART_DIR / "calibrator.pkl", "wb") as fh:
    pickle.dump(calib, fh)
with open(ART_DIR / "run_config.json", "w") as fh:
    json.dump({"CFG": CFG, "BEST": BEST, "FEATS": FEATS, "best_iters": best_iters, "oof_macro_f05": score},
              fh, indent=2, default=str)
imp = pd.Series(np.mean([m.feature_importance("gain") for m in models], axis=0), index=FEATS)
print("Top 15 features by gain:\n", (imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_string())
tick("ALL DONE")
