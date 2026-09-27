# %% [markdown]
# # Business Entity Resolution — Amazon ML Challenge 2026 (large-scale version)
#
# Built for the real data volume (~1.5–2M Source 1 rows, ~4–5M rows each in Source 2/3).
#
# **Pipeline:** Arrow-vectorised normalisation → word 1-2-gram TF-IDF keys → inverted-index
# blocking with frequent-token pruning (top-K per source, K chosen from the training recall
# curve) → batched similarity features → LightGBM → threshold tuned on exact macro F0.5 →
# one-to-one resolution → TSV outputs.
#
# Rules kept: no external data / APIs / geocoding; `country` is an open set of strings (only
# used for an equality feature, never one-hot or filtered), so France works like US/India.

# %%
from __future__ import annotations

import gc
import json
import math
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein
import lightgbm as lgb

try:  # optional, multithreaded sparse top-n matmul (MIT licence); scipy fallback otherwise
    from sparse_dot_topn import sp_matmul_topn
    HAVE_SDT = True
except Exception:  # pragma: no cover
    HAVE_SDT = False

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:7.0f}s] {msg}", flush=True)


@dataclass
class Config:
    data_dir: str = "dataset"
    output_dir: str = "output"
    model_dir: str = "models"
    n_threads: int = field(default_factory=lambda: os.cpu_count() or 1)

    # ---- blocking ----
    name_weight: float = 0.6          # block score = 0.6*name_cos + 0.4*addr_cos (word 1-2-gram TF-IDF)
    k_max_per_source: int = 20        # K searched on the training sample
    k_per_source: int | None = None   # None = smallest K keeping `recall_keep` of recall@k_max
    recall_keep: float = 0.995
    max_posting: int = 3000           # tokens in more target records than this are not used for retrieval
    rerank_pool: int = 100            # retrieve this many per source with the pruned index, re-rank by full cosine
    query_chunk: int = 50_000
    cost_budget: int = 40_000_000     # scipy fallback: max multiply-adds per chunk (bounds memory)
    k_name_pass: int = 0              # extra name-only retrieval pass: top-k per source by name cosine (0 = off)
    k_addr_pass: int = 0              # extra address-only retrieval pass: top-k per source by address cosine (0 = off)
    extra_pool: int = 100             # retrieval pool of the extra passes
    indic_translit: bool = False      # romanise Indic-script text (Unicode character names, stdlib only)
    name_skeleton: bool = False       # add a phonetic-skeleton key segment to the name block
    name_clean: bool = False          # names: leet digits -> letters (b1ack -> black), drop web tokens (com, www)
    cross_features: bool = False      # similarity of each candidate to the entity's other top candidates (anchors)
    n_anchors: int = 5
    skip_test: bool = False           # stop after training / validation (quick model checks)
    two_stage: bool = False           # stage-2 model on top of out-of-fold stage-1 probabilities (entity context)
    oof_folds: int = 2
    house_fuzzy: bool = False         # house-number agreement tolerant to truncation / zero padding
    target_features: bool = False     # block ALL train S1 (as at test) + target-side competition features

    # ---- training ----
    train_query_sample: int | None = 200_000   # S1 train entities used as queries (None = all)
    val_frac: float = 0.2
    neg_per_pos: int = 5
    hard_neg_frac: float = 0.6
    max_train_rows: int = 3_000_000
    fast_features: bool = False       # drop the 3 slowest string features
    feature_batch: int = 1_000_000
    seed: int = 42
    lgb_params: dict = field(default_factory=lambda: dict(
        objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
        max_bin=127, verbose=-1))
    num_boost_round: int = 1500
    early_stopping: int = 50

    # ---- decision ----
    threshold_grid: list = field(default_factory=lambda: [round(x, 2) for x in np.arange(0.20, 0.98, 0.01)])
    fixed_threshold: float | None = None
    one_to_one: str = "auto"          # auto | on | off


# %% [markdown]
# ## Step 1 — Loading & preprocessing (vectorised with pyarrow)
# * Multithreaded TSV reader, tab delimiter, **no quote handling** (so a stray `"` can't
#   swallow columns), empty fields become `""`.
# * Normalisation: NFKD + strip accents, lowercase, `&`→`and`, non-alphanumerics→space,
#   Indian PIN `560 001`→`560001`.
# * Tokens are dictionary-encoded once for S1+S2+S3 together; abbreviation expansion
#   (Rd→road, Pvt→private …) is applied to the *vocabulary*, not per row, so it is ~free.
# * Per record we keep: normalised name / core name (legal forms removed) / address text,
#   binary word-unigram + bigram sets for name and address, postal code, house number,
#   country code. The combined record vector is the name block ⊕ address block.

# %%
ABBREV = {
    "corp": "corporation", "inc": "incorporated", "co": "company", "ltd": "limited",
    "pvt": "private", "pvte": "private", "intl": "international", "mfg": "manufacturing",
    "svcs": "services", "svc": "services", "bros": "brothers", "assoc": "associates",
    "mgmt": "management", "tech": "technologies", "grp": "group", "ent": "enterprises",
    "natl": "national", "dept": "department",
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane", "hwy": "highway",
    "pkwy": "parkway", "ct": "court", "pl": "place", "sq": "square", "ste": "suite",
    "apt": "apartment", "fl": "floor", "bldg": "building", "blk": "block", "sec": "sector",
    "opp": "opposite", "nr": "near", "mkt": "market", "stn": "station", "ngr": "nagar",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "no": "number", "num": "number",
}
LEGAL = {
    "incorporated", "corporation", "company", "limited", "private", "llc", "llp", "plc",
    "lp", "pc", "pllc", "gmbh", "sa", "sas", "sasu", "sarl", "eurl", "snc", "the", "and",
    "of", "opc", "ltda",
}
WEB_TOKENS = {"com", "www", "http", "https", "net", "org"}
_LEET = str.maketrans("01345678", "oleasgtb")
_LEET_RE = re.compile(r"(?=(?:.*[a-z]){3})(?!(?:.*\d){3})[a-z0-9]+")


def _clean_name_token(v: str) -> str:
    if v in WEB_TOKENS:
        return ""
    if not v.isalpha() and not v.isdigit() and _LEET_RE.fullmatch(v):
        v = v.translate(_LEET)
    return ABBREV.get(v, v)


def _indic_table() -> dict:
    """str.translate table romanising Indic scripts (U+0900-U+0DFF) from Unicode character names.
    Consonants carry an inherent 'a'; vowel signs / virama emit a \\x01 marker that deletes the
    preceding inherent 'a' (see _translit_indic)."""
    import unicodedata
    vowels = {"A": "a", "AA": "aa", "I": "i", "II": "ii", "U": "u", "UU": "uu", "E": "e", "EE": "ee", "AI": "ai",
              "O": "o", "OO": "oo", "AU": "au", "VOCALIC R": "ri", "VOCALIC RR": "ri", "VOCALIC L": "li",
              "VOCALIC LL": "li", "SHORT E": "e", "SHORT O": "o", "CANDRA E": "e", "CANDRA O": "o", "SHORT A": "a"}
    tab = {0x200C: "", 0x200D: ""}
    for cp in range(0x0900, 0x0E00):
        try:
            nm = unicodedata.name(chr(cp))
        except ValueError:
            continue
        rest = nm.split(" ", 1)[1] if " " in nm else nm
        if rest.startswith("LETTER "):
            x = rest[7:]
            if x in vowels:
                tab[cp] = vowels[x]
            elif x.startswith("CHILLU "):
                tab[cp] = x[7:].lower()
            else:
                tab[cp] = x.lower().replace(" ", "")      # consonant + inherent a ('KHA' -> 'kha')
        elif rest.startswith("VOWEL SIGN "):
            tab[cp] = "\x01" + vowels.get(rest[11:], rest[11:].lower().replace(" ", ""))
        elif rest.endswith("SIGN VIRAMA"):
            tab[cp] = "\x01"
        elif rest.endswith(("ANUSVARA", "CANDRABINDU")):
            tab[cp] = "n"
        elif rest.endswith("VISARGA"):
            tab[cp] = "h"
        elif rest.startswith("DIGIT "):
            tab[cp] = str(unicodedata.digit(chr(cp)))
        elif "LENGTH MARK" in rest or rest.endswith(("NUKTA", "AVAGRAHA")):
            tab[cp] = ""
        else:
            tab[cp] = " "
    return tab


_INDIC_RE = "[\u0900-\u0DFF]"


def _translit_indic(a: pa.Array) -> pa.Array:
    mask = pc.match_substring_regex(a, _INDIC_RE)
    idx = np.flatnonzero(mask.to_numpy(zero_copy_only=False))
    if not len(idx):
        return a
    tab = _indic_table()
    vals = [s.translate(tab).replace("a\x01", "").replace("\x01", "") for s in a.take(pa.array(idx)).to_pylist()]
    log(f"  romanised {len(idx):,} Indic-script values")
    return pc.replace_with_mask(a, mask, pa.array(vals, pa.large_string()))


_SKEL_SUBS = [("x", "ks"), ("ph", "f"), ("sh", "s"), ("ch", "k"), ("ck", "k"), ("ce", "se"), ("ci", "si"), ("q", "k"), ("c", "k"), ("z", "s"),
              ("w", "v"), ("y", "i"), ("d", "t"), ("b", "p"), ("g", "k"), ("j", "k")]
_SKEL_H = re.compile(r"(?<=.)h")
_SKEL_V = re.compile(r"(?<=.)[aeiou]")
_SKEL_DUP = re.compile(r"(.)\1+")


def skeleton(w: str) -> str:
    """Transliteration-insensitive phonetic key of one token ('' = no key)."""
    if len(w) < 2 or not w.isalpha():
        return ""
    for a_, b_ in _SKEL_SUBS:
        w = w.replace(a_, b_)
    w = _SKEL_DUP.sub(r"\1", _SKEL_V.sub("", _SKEL_H.sub("", w)))
    return w if len(w) >= 2 else ""


_POSTAL_RE = re.compile(r"\d{5,6}")
_HOUSE_RE = re.compile(r"\d+[a-z]?")


def read_tsv(path: Path) -> pa.Table:
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
    try:
        return pacsv.read_csv(
            path, read_options=pacsv.ReadOptions(block_size=1 << 26),
            parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
            convert_options=pacsv.ConvertOptions(column_types={c: pa.large_string() for c in header},
                                                 strings_can_be_null=False))
    except pa.ArrowInvalid as e:   # e.g. ragged rows: fall back to pandas
        log(f"pyarrow reader failed on {path.name} ({str(e)[:120]}); using pandas")
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False).fillna("")
        return pa.Table.from_pandas(df, preserve_index=False)


def col(t: pa.Table, name: str) -> pa.Array:
    if name not in t.column_names:
        return pa.array([""] * t.num_rows, pa.large_string())
    return pc.cast(t.column(name).combine_chunks(), pa.large_string())


def normalize(a: pa.Array, is_addr: bool = False) -> pa.Array:
    a = pc.utf8_normalize(a, "NFKD")
    a = pc.replace_substring_regex(a, r"\p{Mn}+", "")
    a = pc.utf8_lower(a)
    a = pc.replace_substring(a, "&", " and ")
    a = pc.replace_substring_regex(a, r"[^\p{L}\p{N}]+", " ")
    if is_addr:
        a = pc.replace_substring_regex(a, r"\b(\d{3}) (\d{3})\b", r"\1\2")
    return pc.utf8_trim_whitespace(a)


def _tokens(norm: pa.Array, mapping: dict | None):
    """Split -> dictionary-encode -> map vocabulary through `mapping` -> drop empty tokens."""
    L = pc.split_pattern(norm, " ")
    par = pc.list_parent_indices(L).to_numpy().astype(np.int32)
    enc = pc.dictionary_encode(pc.list_flatten(L))
    vocab = enc.dictionary.to_pylist()
    if callable(mapping):
        vocab = [mapping(v) for v in vocab]
    elif mapping:
        vocab = [mapping.get(v, v) for v in vocab]
    new_codes, uniq = pd.factorize(np.array(vocab, dtype=object))
    codes = new_codes[enc.indices.to_numpy()].astype(np.int32)
    uniq = np.asarray(uniq, dtype=object)
    empty = np.flatnonzero(uniq == "")
    if len(empty):
        keep = codes != empty[0]
        codes, par = codes[keep], par[keep]
    return codes, par, uniq


def _offsets(par: np.ndarray, n: int) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(np.bincount(par, minlength=n))]).astype(np.int64)


def _rejoin(codes, par, vocab_arrow, n) -> pa.Array:
    lists = pa.LargeListArray.from_arrays(pa.array(_offsets(par, n)), vocab_arrow.take(pa.array(codes)))
    return pc.binary_join(lists, pa.scalar(" ", pa.large_string()))


def _csr_binary(codes, par, n, ncols):
    # copy: csr_matrix keeps a reference to `codes` and sum_duplicates() sorts/compacts it in place,
    # which would shift every later token onto the wrong row for all callers that reuse `codes`
    X = sp.csr_matrix((np.ones(len(codes), np.float32), codes.copy(), _offsets(par, n)), shape=(n, ncols))
    X.sum_duplicates()
    X.data[:] = 1.0
    return X


def _bigrams(codes, par, V):
    same = par[1:] == par[:-1]
    key = codes[:-1][same].astype(np.int64) * V + codes[1:][same]
    bcodes, uniq = pd.factorize(key)
    return bcodes.astype(np.int32), par[:-1][same], len(uniq)


def _tfidf_rows(X: sp.csr_matrix) -> sp.csr_matrix:
    """Binary -> IDF weights -> L2-normalised rows (in place)."""
    df = np.bincount(X.indices, minlength=X.shape[1])
    idf = (np.log((X.shape[0] + 1) / (df + 1)) + 1).astype(np.float32)
    X.data = idf[X.indices]
    sq = np.concatenate([[0.0], np.cumsum(X.data.astype(np.float64) ** 2)])
    norms = np.sqrt(sq[X.indptr[1:]] - sq[X.indptr[:-1]])
    norms[norms == 0] = 1.0
    X.data /= np.repeat(norms, np.diff(X.indptr)).astype(np.float32)
    return X


def _last_per_row(pos_mask, codes, par, n):
    idx = np.flatnonzero(pos_mask)
    out = np.full(n, -1, np.int32)
    if len(idx):
        rows = par[idx]
        last = np.r_[rows[1:] != rows[:-1], True]
        out[rows[last]] = codes[idx[last]]
    return out


def build_field(raw: pa.Array, n: int, kind: str, cfg: Config | None = None) -> dict:
    """Tokenise one text field for all records; return its TF-IDF block and helpers."""
    cfg = cfg or Config()
    if cfg.indic_translit:
        raw = _translit_indic(raw)
    norm = normalize(raw, is_addr=(kind == "addr"))
    codes, par, vocab = _tokens(norm, _clean_name_token if (kind == "name" and cfg.name_clean) else ABBREV)
    V = len(vocab)
    vocab_arrow = pa.array(vocab.tolist(), pa.large_string())
    U = _csr_binary(codes, par, n, V)
    bc, bp, nb = _bigrams(codes, par, V)
    Bg = _csr_binary(bc, bp, n, nb)
    out = {"n_uni": V, "n_bi": nb, "seg_sizes": [V, nb],
           "cnts": [np.diff(U.indptr).astype(np.int32), np.diff(Bg.indptr).astype(np.int32)]}
    blocks = [U, Bg]
    if kind == "name" and cfg.name_skeleton:
        sk_codes, sk_vocab = pd.factorize(np.array([skeleton(v) for v in vocab], dtype=object))
        sk_empty = np.flatnonzero(np.asarray(sk_vocab, dtype=object) == "")
        sc = sk_codes[codes]
        keep = sc != (sk_empty[0] if len(sk_empty) else -1)
        S = _csr_binary(sc[keep].astype(np.int32), par[keep], n, len(sk_vocab))
        out["seg_sizes"].append(len(sk_vocab))
        out["cnts"].append(np.diff(S.indptr).astype(np.int32))
        blocks.append(S)
        log(f"  name skeleton keys: {len(sk_vocab):,}")
    X = sp.hstack(blocks, format="csr", dtype=np.float32)
    del U, Bg, bc, bp, blocks
    out["X"] = _tfidf_rows(X)
    out["text"] = _rejoin(codes, par, vocab_arrow, n)
    out["len"] = pc.utf8_length(out["text"]).to_numpy(zero_copy_only=False).astype(np.int32)
    if kind == "name":
        legal = np.isin(vocab, list(LEGAL))
        keep = ~legal[codes]
        cc, cp = codes[keep], par[keep]
        core = _rejoin(cc, cp, vocab_arrow, n)
        has_core = np.bincount(cp, minlength=n) > 0
        out["core"] = pc.if_else(pa.array(has_core), core, out["text"])
        first = np.full(n, -1, np.int32)
        off = _offsets(cp, n)
        first[has_core] = cc[off[:-1][has_core]]
        out["first"] = first
    else:
        is_postal = np.array([bool(_POSTAL_RE.fullmatch(v)) for v in vocab])
        is_house = np.array([bool(_HOUSE_RE.fullmatch(v)) for v in vocab])
        out["postal"] = _last_per_row(is_postal[codes], codes, par, n)
        off = _offsets(par, n)
        has = np.diff(off) > 0
        first_codes = np.full(n, -1, np.int32)
        first_codes[has] = codes[off[:-1][has]]
        out["house"] = np.where((first_codes >= 0) & is_house[np.maximum(first_codes, 0)], first_codes, -1)
        num_val = np.array([int(m.group()[:9]) if (m := re.match(r"\d+", v)) else -1 for v in vocab], np.int64)
        idx = np.flatnonzero(num_val[codes] >= 0)
        hn = np.full(n, -1, np.int64)
        if len(idx):
            rows = par[idx]
            firsts = np.r_[True, rows[1:] != rows[:-1]]
            hn[rows[firsts]] = num_val[codes[idx[firsts]]]
        out["house_num"] = hn
    return out


@dataclass
class Split:
    name: str
    n1: int
    nT: int
    ids: pa.Array            # S1 ids then S2 ids then S3 ids
    src: np.ndarray          # 0 = S2, 1 = S3 (length nT)
    M: sp.csr_matrix         # [sqrt(w)*name block | sqrt(1-w)*addr block], rows = S1 then T
    bounds: np.ndarray       # column boundaries: name_uni | name_bi | addr_uni | addr_bi
    seg_cnt: np.ndarray      # (n, 4) number of distinct keys per segment
    name_text: pa.Array
    core_text: pa.Array
    addr_text: pa.Array
    name_len: np.ndarray
    addr_len: np.ndarray
    country: np.ndarray
    postal: np.ndarray
    house: np.ndarray
    first: np.ndarray
    country_labels: np.ndarray
    n_name_seg: int = 2              # number of name segments at the start of M (the rest is address)
    gt_q: np.ndarray | None = None   # ground-truth pairs as (S1 row, T row)
    gt_t: np.ndarray | None = None


def prepare_split(cfg: Config, split: str) -> Split:
    d = Path(cfg.data_dir) / split
    tabs = [read_tsv(d / f"{split}_source{i}.tsv") for i in (1, 2, 3)]
    n1, n2, n3 = (t.num_rows for t in tabs)
    n = n1 + n2 + n3
    log(f"[{split}] loaded S1={n1:,} S2={n2:,} S3={n3:,}")
    cat = lambda c: pa.concat_arrays([col(t, c) for t in tabs])
    ids = pc.utf8_trim_whitespace(cat("entity_id"))
    name_raw, addr_raw, ctry_raw = cat("business_name"), cat("business_address"), cat("country")
    del tabs
    nf = build_field(name_raw, n, "name", cfg); del name_raw
    log(f"[{split}] name field: {nf['n_uni']:,} words, {nf['n_bi']:,} bigrams")
    af = build_field(addr_raw, n, "addr", cfg); del addr_raw
    log(f"[{split}] addr field: {af['n_uni']:,} words, {af['n_bi']:,} bigrams")
    w = cfg.name_weight
    M = sp.hstack([nf.pop("X") * np.float32(math.sqrt(w)), af.pop("X") * np.float32(math.sqrt(1 - w))],
                  format="csr", dtype=np.float32)
    bounds = np.cumsum([0] + nf["seg_sizes"] + af["seg_sizes"])
    cn = pc.dictionary_encode(normalize(ctry_raw))
    labels = np.asarray(cn.dictionary.to_pylist(), dtype=object)
    country = cn.indices.to_numpy().astype(np.int32)
    if (labels == "").any():
        country[country == np.flatnonzero(labels == "")[0]] = -1
    D = Split(split, n1, n2 + n3, ids, np.r_[np.zeros(n2, np.int8), np.ones(n3, np.int8)], M, bounds,
              np.column_stack(nf["cnts"] + af["cnts"]),
              nf["text"], nf["core"], af["text"], nf["len"], af["len"], country,
              af["postal"], af["house"], nf["first"], labels, len(nf["seg_sizes"]))
    D.house_num = af["house_num"]
    gt_path = d / f"{split}_ground_truth.tsv"
    if gt_path.exists():
        g = read_tsv(gt_path)
        L = pc.split_pattern(col(g, "matched_entity_ids"), ",")
        par = pc.list_parent_indices(L)
        tid = pc.utf8_trim_whitespace(pc.list_flatten(L))
        sid = col(g, "source1_entity_id").take(par)
        q = pc.index_in(pc.utf8_trim_whitespace(sid), value_set=ids.slice(0, n1)).to_numpy(zero_copy_only=False)
        t = pc.index_in(tid, value_set=ids.slice(n1)).to_numpy(zero_copy_only=False)
        ok = ~(np.isnan(q.astype(float)) | np.isnan(t.astype(float)))
        D.gt_q, D.gt_t = q[ok].astype(np.int32), t[ok].astype(np.int32)
    gc.collect()
    log(f"[{split}] prepared: M={M.shape[0]:,}x{M.shape[1]:,} nnz={M.nnz:,}, "
        f"countries={labels.tolist()[:8]}")
    return D


def profile(D: Split) -> dict:
    """Quick stats that drive K, sampling and the one-to-one rule."""
    per_s1 = np.bincount(D.gt_q, minlength=D.n1)
    per_t = np.bincount(D.gt_t, minlength=D.nT)
    per_src = [np.bincount(D.gt_q[D.src[D.gt_t] == s], minlength=D.n1) for s in (0, 1)]
    st = {"s1": D.n1, "targets": D.nT, "gt_pairs": int(len(D.gt_q)),
          "singleton_share": float((per_s1 == 0).mean()), "mean_matches": float(per_s1.mean()),
          "p99_matches_per_source": [int(np.percentile(x, 99)) for x in per_src],
          "max_matches_per_source": [int(x.max()) for x in per_src],
          "targets_matched_share": float((per_t > 0).mean()),
          "targets_with_2plus_s1": float((per_t > 1).mean())}
    for k, v in st.items():
        log(f"  profile {k}: {v}")
    return st


# %% [markdown]
# ## Step 2 — Blocking
# Score = cosine of record vectors (0.6·name + 0.4·address word-1-2-gram TF-IDF). Two stages:
# 1. **Retrieve** a wide pool (`rerank_pool`, default 100 per source) with a sparse product
#    against an inverted index where tokens occurring in more than `max_posting` target
#    records are removed ("road", "limited", big-city names…). Cost ≈ linear in data size.
#    `sparse_dot_topn` runs it multithreaded if installed; chunked scipy fallback otherwise.
# 2. **Re-rank** the pool by the *full, unpruned* cosine and keep the top-K per source.
#    Pruned scores ignore common words, so without this step true matches tie with junk;
#    on the benchmark this raised recall@20 from 0.73 to 0.85 for ~2x the blocking time.

# %%
def _topk_rows(C: sp.csr_matrix, k: int):
    lens = np.diff(C.indptr)
    rows = np.repeat(np.arange(C.shape[0]), lens)
    order = np.lexsort((-C.data, rows))
    rank = np.arange(len(order)) - np.repeat(C.indptr[:-1], lens)
    keep = order[rank < k]
    return rows[keep], C.indices[keep], C.data[keep]


def rowwise_dot(M: sp.csr_matrix, ia: np.ndarray, ib: np.ndarray, batch: int = 500_000) -> np.ndarray:
    """Full (unpruned) cosine for explicit pairs of rows of M."""
    out = np.empty(len(ia), np.float32)
    for s in range(0, len(ia), batch):
        out[s:s + batch] = np.asarray(M[ia[s:s + batch]].multiply(M[ib[s:s + batch]]).sum(axis=1)).ravel()
    return out


def _retrieve(Qc: sp.csr_matrix, BT: sp.csr_matrix, dfp: np.ndarray, pool: int, cfg: Config):
    """Top-`pool` targets per query row by pruned-index score. Returns (row, target col)."""
    if HAVE_SDT:
        C = sp_matmul_topn(Qc, BT, top_n=pool, threshold=0.0, sort=False,
                           n_threads=cfg.n_threads if cfg.n_threads > 1 else None)
        return np.repeat(np.arange(C.shape[0]), np.diff(C.indptr)), C.indices.copy()
    cum = np.cumsum(np.add.reduceat(dfp[Qc.indices], Qc.indptr[:-1]) * (np.diff(Qc.indptr) > 0)) if Qc.nnz else np.zeros(Qc.shape[0])
    cuts = np.unique(np.r_[0, np.searchsorted(cum, np.arange(cfg.cost_budget, cum[-1] + cfg.cost_budget, cfg.cost_budget)) + 1, Qc.shape[0]])
    cuts = cuts[cuts <= Qc.shape[0]]
    rs, cs = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        r, c, _ = _topk_rows((Qc[a:b] @ BT).tocsr(), pool)
        rs.append(r + a); cs.append(c)
    return (np.concatenate(rs), np.concatenate(cs)) if rs else (np.array([], int), np.array([], int))


def rowwise_dot_parts(M: sp.csr_matrix, ia: np.ndarray, ib: np.ndarray, split_col: int, batch: int = 500_000):
    """Full cosine contributions of the columns < split_col (name) and >= split_col (address)."""
    nm = np.empty(len(ia), np.float32)
    ad = np.empty(len(ia), np.float32)
    for s in range(0, len(ia), batch):
        P = M[ia[s:s + batch]].multiply(M[ib[s:s + batch]]).tocsr()
        rows = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr))
        is_n = P.indices < split_col
        nm[s:s + batch] = np.bincount(rows[is_n], weights=P.data[is_n], minlength=P.shape[0])
        ad[s:s + batch] = np.bincount(rows[~is_n], weights=P.data[~is_n], minlength=P.shape[0])
    return nm, ad


def _mask_cols(X: sp.csr_matrix, keep_lo: int, keep_hi: int) -> sp.csr_matrix:
    X.data[(X.indices < keep_lo) | (X.indices >= keep_hi)] = 0
    X.eliminate_zeros()
    return X


PASSES = {"all": 1, "name": 2, "addr": 4}


def search_source(D: Split, q_rows: np.ndarray, src: int, k: int, cfg: Config, part: str = "all"):
    """Pruned inverted-index retrieval of a wide pool, then exact re-rank -> top-k.
    part='all' ranks by the full (name+address) cosine; 'name' / 'addr' retrieve and rank using only
    that field's columns. Returns (q, t, full cosine, rank within this pass)."""
    t_rows = np.flatnonzero(D.src == src).astype(np.int32)
    split_col = int(D.bounds[D.n_name_seg])
    lo, hi = {"all": (0, D.M.shape[1]), "name": (0, split_col), "addr": (split_col, D.M.shape[1])}[part]
    Q = D.M[q_rows]
    B = D.M[D.n1 + t_rows]
    if part != "all":
        Q, B = _mask_cols(Q, lo, hi), _mask_cols(B, lo, hi)
    df = np.bincount(B.indices, minlength=B.shape[1])
    B.data[df[B.indices] > cfg.max_posting] = 0
    B.eliminate_zeros()
    dfp = np.bincount(B.indices, minlength=B.shape[1]).astype(np.int64)
    cost = int(dfp[Q.indices].sum())
    BT = B.T.tocsr()
    del B
    pool = max(k, cfg.rerank_pool if part == "all" else cfg.extra_pool)
    t0 = time.time()
    qs, ts, ss, rs = [], [], [], []
    for s in range(0, len(q_rows), cfg.query_chunk):
        r, c = _retrieve(Q[s:s + cfg.query_chunk], BT, dfp, pool, cfg)
        qq, tt = q_rows[s + r], t_rows[c]
        nm, ad = rowwise_dot_parts(D.M, qq, D.n1 + tt, split_col)
        full = nm + ad
        ps = {"all": full, "name": nm, "addr": ad}[part]
        rk, _ = _ranks(qq.astype(np.int64), ps)
        m = (rk <= k) & (ps > 0)
        qs.append(qq[m]); ts.append(tt[m]); ss.append(full[m]); rs.append(rk[m])
    del Q, BT
    log(f"  S{src + 2} [{part}]: {len(q_rows):,} queries x {len(t_rows):,} targets, {cost / 1e9:.2f}G index ops, "
        f"pool {pool} -> top {k}: {time.time() - t0:.0f}s")
    return (np.concatenate(qs).astype(np.int32), np.concatenate(ts).astype(np.int32),
            np.concatenate(ss).astype(np.float32), np.concatenate(rs).astype(np.float32))


def _union_passes(runs: list, nT: int) -> dict:
    """Union (q, t) pairs over passes: OR of pass bits, min rank per pass (NaN = not retrieved).
    runs: list of (q, t, full score, pass rank, pass name)."""
    key = np.concatenate([r[0].astype(np.int64) * nT + r[1] for r in runs])
    order = np.argsort(key, kind="stable")
    key = key[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    out = {"q": (key[starts] // nT).astype(np.int32), "t": (key[starts] % nT).astype(np.int32)}
    del key
    out["block_score"] = np.concatenate([r[2] for r in runs])[order][starts]
    bits = np.concatenate([np.full(len(r[0]), PASSES[r[4]], np.int8) for r in runs])[order]
    out["pass"] = np.bitwise_or.reduceat(bits, starts).astype(np.int8)
    del bits
    for name, bit in PASSES.items():
        rk = np.concatenate([r[3] if PASSES[r[4]] == bit else np.full(len(r[0]), np.inf, np.float32)
                             for r in runs])[order]
        v = np.minimum.reduceat(rk, starts)
        out[f"pass_rank_{name}"] = np.where(np.isinf(v), np.nan, v).astype(np.float32)
    return out


def _ranks(keys_major: np.ndarray, score: np.ndarray):
    """Rank (1 = best) of each row by score within groups of keys_major, plus group best."""
    order = np.lexsort((-score, keys_major))
    km = keys_major[order]
    start = np.r_[True, km[1:] != km[:-1]]
    grp_start = np.maximum.accumulate(np.where(start, np.arange(len(km)), 0))
    rank = np.empty(len(km), np.float32)
    rank[order] = np.arange(len(km)) - grp_start + 1
    best = np.empty(len(km), np.float32)
    best[order] = score[order][grp_start]
    return rank, best


def add_group_features(C: pd.DataFrame) -> pd.DataFrame:
    q, s = C["q"].to_numpy().astype(np.int64), C["block_score"].to_numpy()
    C["rank_in_source"], best = _ranks(q * 2 + C["src"].to_numpy(), s)
    C["rank_overall"], _ = _ranks(q, s)
    C["gap_to_best"] = (best - s).astype(np.float32)
    C["ratio_to_best"] = np.where(best > 0, s / np.maximum(best, 1e-9), np.nan).astype(np.float32)
    C["n_cand_s1"] = np.bincount(q)[q].astype(np.float32)
    return C


def generate_candidates(D: Split, q_rows: np.ndarray, k: int, cfg: Config) -> pd.DataFrame:
    """Per source: combined-cosine pass (top k) + optional name-only / address-only passes, unioned."""
    t0 = time.time()
    parts = []
    passes = [("all", k), ("name", cfg.k_name_pass), ("addr", cfg.k_addr_pass)]
    for src in (0, 1):
        runs = []
        for part, kk in passes:
            if kk > 0:
                runs.append(search_source(D, q_rows, src, kk, cfg, part) + (part,))
                gc.collect()
        U = _union_passes(runs, D.nT)
        del runs
        parts.append(pd.DataFrame({"src": np.int8(src), **U}))
        del U
        gc.collect()
    C = pd.concat(parts, ignore_index=True)
    del parts
    C = add_group_features(C)
    log(f"[{D.name}] candidates: {len(C):,} ({len(C) / max(len(q_rows), 1):.1f} per S1) in {time.time() - t0:.0f}s")
    return C


def truncate_k(C: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep the top k of the combined pass; candidates of the extra passes are kept as they are."""
    keep = (C["pass_rank_all"] <= k) | C["pass_rank_name"].notna() | C["pass_rank_addr"].notna()
    C = C[keep.to_numpy()].drop(columns=["rank_in_source", "rank_overall", "gap_to_best",
                                         "ratio_to_best", "n_cand_s1"]).reset_index(drop=True)
    C.loc[C["pass_rank_all"] > k, "pass_rank_all"] = np.nan
    C["pass"] = ((C["pass_rank_all"].notna().to_numpy() * 1) | (C["pass_rank_name"].notna().to_numpy() * 2)
                 | (C["pass_rank_addr"].notna().to_numpy() * 4)).astype(np.int8)
    return add_group_features(C)


# %% [markdown]
# ## Step 3 — Labels, K selection & negative sampling
# The training queries are a **random sample of Source 1**, searched against the **full**
# S2/S3, so candidate difficulty matches the test set. The recall-vs-K curve picks the
# smallest K that keeps 99.5 % of the reachable matches — fewer pairs = faster everything.

# %%
def label_pairs(D: Split, q: np.ndarray, t: np.ndarray) -> np.ndarray:
    gt_keys = D.gt_q.astype(np.int64) * D.nT + D.gt_t
    return np.isin(q.astype(np.int64) * D.nT + t, gt_keys).astype(np.int8)


def choose_k(C: pd.DataFrame, D: Split, q_rows: np.ndarray, cfg: Config) -> int:
    in_q = np.isin(D.gt_q, q_rows)
    total = int(in_q.sum())
    pr = C.loc[C["label"] == 1, "pass_rank_all"].fillna(np.inf).to_numpy()
    curve = {k: float((pr <= k).sum() / max(total, 1)) for k in range(1, cfg.k_max_per_source + 1)}
    top = curve[cfg.k_max_per_source]
    k = next(k for k, r in curve.items() if r >= cfg.recall_keep * top)
    log("  recall@K: " + "  ".join(f"{k}:{r:.4f}" for k, r in curve.items() if k in (1, 2, 3, 5, 8, 10, 15, 20) or k == cfg.k_max_per_source))
    log(f"  chosen K per source = {k} (recall {curve[k]:.4f} vs {top:.4f} at K={cfg.k_max_per_source})")
    lab = C["label"].to_numpy() == 1
    extra = C["pass_rank_name"].notna().to_numpy() | C["pass_rank_addr"].notna().to_numpy()
    log(f"  union recall with extra passes (K={k}): "
        f"{((lab & ((C['pass_rank_all'].to_numpy() <= k) | extra)).sum()) / max(total, 1):.4f}")
    return k


def sample_rows(C: pd.DataFrame, cfg: Config, rng) -> np.ndarray:
    pos = np.flatnonzero(C["label"].to_numpy() == 1)
    neg = np.flatnonzero(C["label"].to_numpy() == 0)
    n_neg = min(len(neg), cfg.neg_per_pos * max(len(pos), 1))
    hard_mask = C["rank_overall"].to_numpy()[neg] <= 3
    hard, easy = neg[hard_mask], neg[~hard_mask]
    n_hard = min(len(hard), int(n_neg * cfg.hard_neg_frac))
    n_easy = min(len(easy), n_neg - n_hard)
    idx = np.concatenate([pos, rng.choice(hard, n_hard, replace=False), rng.choice(easy, n_easy, replace=False)])
    if len(idx) > cfg.max_train_rows:
        idx = rng.choice(idx, cfg.max_train_rows, replace=False)
    log(f"  sampled {len(idx):,} rows ({int(C['label'].to_numpy()[idx].sum()):,} positives)")
    return np.sort(idx)


# %% [markdown]
# ## Step 4 — Features (batched, bounded memory)
# Requested: `name_jaccard`, `name_levenshtein_ratio`, `address_jaccard`,
# `tfidf_cosine_score` (full, unpruned 0.6·name + 0.4·address TF-IDF cosine), `country_match`.
# Extra: bigram Jaccards, per-field TF-IDF cosines, rapidfuzz token-sort / token-set /
# partial ratios, core-name ratio, first-token, postal-code and house-number agreement,
# lengths, source flag, and candidate-context features (rank, gap/ratio to the best candidate).
# Word overlaps come from one sparse element-wise product per batch; string ratios from
# rapidfuzz `cpdist` (C++, multithreaded).

# %%
BASE_FEATURES = [
    "name_jaccard", "name_levenshtein_ratio", "address_jaccard", "tfidf_cosine_score", "country_match",
    "name_bigram_jaccard", "addr_bigram_jaccard", "name_tfidf_cos", "addr_tfidf_cos",
    "name_token_sort", "name_core_ratio", "first_token_match", "addr_token_set",
    "postal_match", "house_match", "name_len_s1", "name_len_t", "addr_len_s1", "addr_len_t",
    "is_s3", "rank_in_source", "rank_overall", "gap_to_best", "ratio_to_best", "n_cand_s1",
    "pass_rank_all", "pass_rank_name", "pass_rank_addr", "name_skel_jaccard",
]
SLOW_FEATURES = ["name_token_set", "name_partial", "addr_levenshtein_ratio"]
CROSS_FEATURES = ["cross_max_full", "cross_mean_full", "cross_max_name", "cross_max_addr", "cross_cnt_05",
                  "cross_is_anchor", "cross_top_name_set", "cross_top_addr_set"]
HOUSE_FEATURES = ["house_fuzzy", "house_logdiff"]
TARGET_FEATURES = ["t_n_s1", "t_rank", "t_gap", "t_ratio", "t_margin"]
P1_COLS = ["p1", "p1_rank", "p1_rank_src", "p1_max_other", "p1_second", "p1_sum_other", "p1_cnt05_other",
           "p1_cnt05_other_src", "p1_ratio_best", "p1_gap_best"]
STAGE2_FEATURES = P1_COLS + ["pc_max_full", "pc_wmean_full", "pc_max_name", "pc_max_addr", "pc_cnt_05",
                             "pc_is_anchor", "pc_top_name_set", "pc_top_addr_set"]


def feature_names(cfg: Config, stage: int = 1) -> list:
    return (BASE_FEATURES + ([] if cfg.fast_features else SLOW_FEATURES)
            + (CROSS_FEATURES if cfg.cross_features else []) + (HOUSE_FEATURES if cfg.house_fuzzy else [])
            + (TARGET_FEATURES if cfg.target_features else [])
            + (STAGE2_FEATURES if stage == 2 else []))


def compute_anchors(C: pd.DataFrame, n1: int, n_anchors: int, score=None, with_scores: bool = False):
    """(n1, n_anchors) target rows of each S1's best candidates by `score` (default block score;
    -1 = none). Must be computed on the full candidate set (before any negative sampling) so train
    and test agree. with_scores=True also returns the anchors' scores."""
    q, t = C["q"].to_numpy(), C["t"].to_numpy()
    sc = C["block_score"].to_numpy() if score is None else score
    order = np.lexsort((-sc, q))
    qs = q[order]
    start = np.r_[True, qs[1:] != qs[:-1]]
    gstart = np.maximum.accumulate(np.where(start, np.arange(len(qs)), 0))
    rank = np.arange(len(qs)) - gstart
    A = np.full((n1, n_anchors), -1, np.int32)
    S = np.zeros((n1, n_anchors), np.float32)
    m = rank < n_anchors
    A[qs[m], rank[m]] = t[order][m]
    S[qs[m], rank[m]] = sc[order][m]
    return (A, S) if with_scores else A


def add_target_features(C: pd.DataFrame) -> pd.DataFrame:
    """Target-side competition: how this S1 compares with every other S1 that retrieved the same
    S2/S3 record. Needs candidates for ALL S1 entities (true at test; --target-features at train)."""
    t = C["t"].to_numpy().astype(np.int64)
    sc = C["block_score"].to_numpy().astype(np.float32)
    rank, best = _ranks(t, sc)
    order = np.lexsort((-sc, t))
    ts, ss = t[order], sc[order]
    start = np.r_[True, ts[1:] != ts[:-1]]
    gs = np.maximum.accumulate(np.where(start, np.arange(len(ts)), 0))
    cnt = np.bincount(t)
    sec_sorted = np.where(cnt[ts] > 1, ss[np.minimum(gs + 1, len(ss) - 1)], 0.0)
    second = np.empty(len(t), np.float32)
    second[order] = sec_sorted
    C["t_n_s1"] = cnt[t].astype(np.float32)
    C["t_rank"] = rank
    C["t_gap"] = (best - sc).astype(np.float32)
    C["t_ratio"] = np.where(best > 0, sc / np.maximum(best, 1e-9), np.nan).astype(np.float32)
    C["t_margin"] = (sc - np.where(rank == 1, second, best)).astype(np.float32)
    return C


def add_p1_features(C: pd.DataFrame, p1: np.ndarray) -> pd.DataFrame:
    """Entity-level context of stage-1 probabilities (vectorised over the full candidate set)."""
    q = C["q"].to_numpy().astype(np.int64)
    src = C["src"].to_numpy().astype(np.int64)
    p1 = p1.astype(np.float32)
    C["p1"] = p1
    C["p1_rank"], best = _ranks(q, p1)
    C["p1_rank_src"], _ = _ranks(q * 2 + src, p1)
    order = np.lexsort((-p1, q))
    qs, ps = q[order], p1[order]
    start = np.r_[True, qs[1:] != qs[:-1]]
    gs = np.maximum.accumulate(np.where(start, np.arange(len(qs)), 0))
    size = np.bincount(q, minlength=q.max() + 1 if len(q) else 1)
    has2 = size[qs] > 1
    sec_sorted = np.where(has2, ps[np.minimum(gs + 1, len(ps) - 1)], 0.0)
    second = np.empty(len(q), np.float32)
    second[order] = sec_sorted
    C["p1_second"] = second
    C["p1_max_other"] = np.where(C["p1_rank"].to_numpy() == 1, second, best).astype(np.float32)
    tot = np.bincount(q, weights=p1)[q]
    C["p1_sum_other"] = (tot - p1).astype(np.float32)
    hi = (p1 > 0.5).astype(np.float64)
    C["p1_cnt05_other"] = (np.bincount(q, weights=hi)[q] - hi).astype(np.float32)
    key_other = q * 2 + (1 - src)
    cnt_qs = np.bincount(q * 2 + src, weights=hi, minlength=int(key_other.max()) + 1 if len(q) else 1)
    C["p1_cnt05_other_src"] = cnt_qs[key_other].astype(np.float32)
    C["p1_ratio_best"] = np.where(best > 0, p1 / np.maximum(best, 1e-9), np.nan).astype(np.float32)
    C["p1_gap_best"] = (best - p1).astype(np.float32)
    return C


def cross_features(D: Split, q: np.ndarray, t: np.ndarray, cfg: Config, wk: int,
                   anchors=None, prefix: str = "cross_", anchor_w=None) -> dict:
    """Agreement of candidate t with the other top candidates of the same S1 (S2/S3 copies of one
    business tend to agree with each other even when the S1 record is noisy). With anchor_w (the
    anchors' stage-1 probabilities) a probability-weighted mean is returned instead of the plain mean."""
    A = (D.anchors if anchors is None else anchors)[q]
    m, k = A.shape
    full = np.full((m, k), np.nan, np.float32)
    nmx = np.full((m, k), np.nan, np.float32)
    adx = np.full((m, k), np.nan, np.float32)
    split_col = int(D.bounds[D.n_name_seg])
    for j in range(k):
        a = A[:, j]
        v = np.flatnonzero((a >= 0) & (a != t))
        if len(v):
            nm, ad = rowwise_dot_parts(D.M, D.n1 + t[v], D.n1 + a[v], split_col)
            full[v, j], nmx[v, j], adx[v, j] = nm + ad, nm / cfg.name_weight, ad / (1 - cfg.name_weight)
    F = {}
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            F["max_full"] = np.nanmax(full, 1)
            if anchor_w is None:
                F["mean_full"] = np.nanmean(full, 1)
            else:
                W = np.where(np.isnan(full), 0.0, anchor_w[q])
                F["wmean_full"] = (np.nan_to_num(full) * W).sum(1) / np.maximum(W.sum(1), 1e-6)
            F["max_name"] = np.nanmax(nmx, 1)
            F["max_addr"] = np.nanmax(adx, 1)
    F["cnt_05"] = (np.nan_to_num(full) > 0.5).sum(1).astype(np.float32)
    F["is_anchor"] = (A == t[:, None]).any(1).astype(np.float32)
    # string similarity to the best other anchor
    best = np.where(A[:, 0] != t, A[:, 0], A[:, 1] if k > 1 else -1)
    has = best >= 0
    tn, bn = np.full(m, np.nan, np.float32), np.full(m, np.nan, np.float32)
    if has.any():
        it, ia = pa.array(D.n1 + t[has]), pa.array(D.n1 + best[has])
        tn[has] = _cp(D.name_text.take(it).to_pylist(), D.name_text.take(ia).to_pylist(), fuzz.token_set_ratio, wk)
        bn[has] = _cp(D.addr_text.take(it).to_pylist(), D.addr_text.take(ia).to_pylist(), fuzz.token_set_ratio, wk)
    F["top_name_set"], F["top_addr_set"] = tn, bn
    return {prefix + c: np.asarray(v, np.float32) for c, v in F.items()}


def _cp(a, b, scorer, workers):
    return process.cpdist(a, b, scorer=scorer, workers=workers).astype(np.float32) / 100.0


def _lev(a, b, **kw):
    return 100.0 * Levenshtein.normalized_similarity(a, b)


def _agree(x, y):
    out = (x == y).astype(np.float32)
    out[(x < 0) | (y < 0)] = np.nan
    return out


def house_fuzzy(h1: np.ndarray, h2: np.ndarray):
    """2 = same number, 1 = one is a prefix / suffix of the other (truncation, e.g. 1344 vs 344),
    0 = different, NaN = missing; plus log1p of the absolute difference."""
    ok = (h1 >= 0) & (h2 >= 0)
    a, b = np.maximum(h1, 0), np.maximum(h2, 0)
    da = np.floor(np.log10(np.maximum(a, 1))).astype(np.int64) + 1
    db = np.floor(np.log10(np.maximum(b, 1))).astype(np.int64) + 1
    suf = (a % (10 ** db) == b) | (b % (10 ** da) == a)
    pre = ((da > db) & (a // (10 ** np.maximum(da - db, 0)) == b)) | ((db > da) & (b // (10 ** np.maximum(db - da, 0)) == a))
    f = np.where(a == b, 2.0, np.where(suf | pre, 1.0, 0.0)).astype(np.float32)
    f[~ok] = np.nan
    d = np.log1p(np.abs(a - b)).astype(np.float32)
    d[~ok] = np.nan
    return f, d


def pair_features(D: Split, C: pd.DataFrame, cfg: Config, stage: int = 1) -> pd.DataFrame:
    q = C["q"].to_numpy(); t = C["t"].to_numpy(); ib = D.n1 + t; m = len(q)
    F = {}
    P = D.M[q].multiply(D.M[ib]).tocsr()
    rows = np.repeat(np.arange(m), np.diff(P.indptr))
    ns, nn = len(D.bounds) - 1, D.n_name_seg
    seg = np.searchsorted(D.bounds, P.indices, side="right") - 1
    key = rows * ns + seg
    sums = np.bincount(key, weights=P.data, minlength=ns * m).reshape(m, ns)
    inter = np.bincount(key, minlength=ns * m).reshape(m, ns).astype(np.float32)
    del P, rows, seg, key
    union = D.seg_cnt[q] + D.seg_cnt[ib] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, inter / union, np.nan).astype(np.float32)
    w = cfg.name_weight
    F["name_tfidf_cos"] = (sums[:, :nn].sum(1) / w).astype(np.float32)
    F["addr_tfidf_cos"] = (sums[:, nn:].sum(1) / (1 - w)).astype(np.float32)
    F["tfidf_cosine_score"] = (sums.sum(1)).astype(np.float32)
    F["name_jaccard"], F["name_bigram_jaccard"] = jac[:, 0], jac[:, 1]
    F["address_jaccard"], F["addr_bigram_jaccard"] = jac[:, nn], jac[:, nn + 1]
    F["name_skel_jaccard"] = jac[:, 2] if nn > 2 else np.full(m, np.nan, np.float32)
    F["country_match"] = _agree(D.country[q], D.country[ib])
    F["postal_match"] = _agree(D.postal[q], D.postal[ib])
    F["house_match"] = _agree(D.house[q], D.house[ib])
    F["first_token_match"] = _agree(D.first[q], D.first[ib])
    F["name_len_s1"], F["name_len_t"] = D.name_len[q].astype(np.float32), D.name_len[ib].astype(np.float32)
    F["addr_len_s1"], F["addr_len_t"] = D.addr_len[q].astype(np.float32), D.addr_len[ib].astype(np.float32)
    wk = cfg.n_threads if cfg.n_threads > 1 else 1
    iq, it = pa.array(q), pa.array(ib)
    n1, n2 = D.name_text.take(iq).to_pylist(), D.name_text.take(it).to_pylist()
    F["name_levenshtein_ratio"] = _cp(n1, n2, _lev, wk)
    F["name_token_sort"] = _cp(n1, n2, fuzz.token_sort_ratio, wk)
    if not cfg.fast_features:
        F["name_token_set"] = _cp(n1, n2, fuzz.token_set_ratio, wk)
        F["name_partial"] = _cp(n1, n2, fuzz.partial_ratio, wk)
    del n1, n2
    F["name_core_ratio"] = _cp(D.core_text.take(iq).to_pylist(), D.core_text.take(it).to_pylist(), fuzz.token_sort_ratio, wk)
    a1, a2 = D.addr_text.take(iq).to_pylist(), D.addr_text.take(it).to_pylist()
    F["addr_token_set"] = _cp(a1, a2, fuzz.token_set_ratio, wk)
    if not cfg.fast_features:
        F["addr_levenshtein_ratio"] = _cp(a1, a2, _lev, wk)
    del a1, a2
    F["is_s3"] = C["src"].to_numpy().astype(np.float32)
    for c in ("block_score", "rank_in_source", "rank_overall", "gap_to_best", "ratio_to_best", "n_cand_s1",
              "pass_rank_all", "pass_rank_name", "pass_rank_addr"):
        F[c] = C[c].to_numpy().astype(np.float32)
    if cfg.cross_features:
        F.update(cross_features(D, q, t, cfg, wk))
    if cfg.house_fuzzy:
        F["house_fuzzy"], F["house_logdiff"] = house_fuzzy(D.house_num[q], D.house_num[ib])
    if cfg.target_features:
        for c in TARGET_FEATURES:
            F[c] = C[c].to_numpy().astype(np.float32)
    if stage == 2:
        for c in P1_COLS:
            F[c] = C[c].to_numpy().astype(np.float32)
        F.update(cross_features(D, q, t, cfg, wk, anchors=D.anchors_p, prefix="pc_", anchor_w=D.anchor_pw))
    return pd.DataFrame({c: F[c] for c in feature_names(cfg, stage)})


def features_batched(D: Split, C: pd.DataFrame, cfg: Config, booster=None, tag="", stage: int = 1) -> np.ndarray | pd.DataFrame:
    """Compute features in batches. With a booster, return probabilities only (never holds
    the full feature matrix in memory)."""
    out = []
    t0 = time.time()
    for s in range(0, len(C), cfg.feature_batch):
        X = pair_features(D, C.iloc[s:s + cfg.feature_batch], cfg, stage)
        out.append(booster.predict(X.to_numpy(np.float32), num_threads=cfg.n_threads).astype(np.float32)
                   if booster is not None else X)
        if (s // cfg.feature_batch) % 5 == 0:
            log(f"  {tag} features {min(s + cfg.feature_batch, len(C)):,}/{len(C):,} ({time.time() - t0:.0f}s)")
    if booster is not None:
        return np.concatenate(out) if out else np.array([], np.float32)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=feature_names(cfg, stage))


# %% [markdown]
# ## Step 5 — LightGBM, exact macro-F0.5 threshold tuning, one-to-one rule
# Validation = 20 % of the sampled S1 entities with their **full** candidate lists. Recall
# denominators use the full ground truth, so blocking misses are counted as misses.
# One-to-one (each S2/S3 record keeps only its best S1) is enabled when the training ground
# truth shows almost no target linked to several S1 entities.

# %%
def target_best(t: np.ndarray, prob: np.ndarray) -> np.ndarray:
    order = np.lexsort((-prob, t))
    first = np.r_[True, t[order][1:] != t[order][:-1]]
    out = np.zeros(len(t), bool)
    out[order[first]] = True
    return out


def macro_f05(nq: int, qi: np.ndarray, is_tp: np.ndarray, gt_cnt: np.ndarray) -> float:
    """qi: query index (0..nq-1) of each predicted pair; is_tp: whether it is a true match."""
    npred = np.bincount(qi, minlength=nq)
    tp = np.bincount(qi, weights=is_tp.astype(np.float64), minlength=nq)
    f = ((npred == 0) & (gt_cnt == 0)).astype(np.float64)
    m = tp > 0
    P, R = tp[m] / npred[m], tp[m] / gt_cnt[m]
    f[m] = 1.25 * P * R / (0.25 * P + R)
    return float(f.mean())


def train_lgb(Xtr, ytr, Xva, yva, cfg: Config, rounds=None):
    params = dict(cfg.lgb_params, num_threads=cfg.n_threads, seed=cfg.seed)
    dtr = lgb.Dataset(Xtr.to_numpy(np.float32), ytr, feature_name=list(Xtr.columns), free_raw_data=True)
    if rounds is not None:
        return lgb.train(params, dtr, num_boost_round=rounds)
    dva = lgb.Dataset(Xva.to_numpy(np.float32), yva, reference=dtr)
    return lgb.train(params, dtr, num_boost_round=cfg.num_boost_round, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(cfg.early_stopping, verbose=False), lgb.log_evaluation(0)])


def train_and_tune(D: Split, cfg: Config) -> dict:
    rng = np.random.default_rng(cfg.seed)
    st = profile(D)
    n_q = D.n1 if cfg.train_query_sample is None else min(D.n1, cfg.train_query_sample)
    q_rows = np.sort(rng.choice(D.n1, n_q, replace=False)).astype(np.int32)
    if cfg.target_features:   # block every S1, exactly like the test set, so target-side features agree
        C = generate_candidates(D, np.arange(D.n1, dtype=np.int32), cfg.k_max_per_source, cfg)
    else:
        C = generate_candidates(D, q_rows, cfg.k_max_per_source, cfg)
    C["label"] = label_pairs(D, C["q"].to_numpy(), C["t"].to_numpy())
    if cfg.k_per_source:
        k = cfg.k_per_source
    else:
        k = choose_k(C[np.isin(C["q"].to_numpy(), q_rows)] if cfg.target_features else C, D, q_rows, cfg)
    C = truncate_k(C, k)
    if cfg.target_features:
        C = add_target_features(C)
        C = C[np.isin(C["q"].to_numpy(), q_rows)].reset_index(drop=True)
        gc.collect()
        log(f"  target-side features computed on all S1; kept {len(C):,} candidates of the {len(q_rows):,} sampled S1")
    C["label"] = label_pairs(D, C["q"].to_numpy(), C["t"].to_numpy())
    if cfg.cross_features:
        D.anchors = compute_anchors(C, D.n1, cfg.n_anchors)

    val_q = np.sort(rng.choice(q_rows, int(len(q_rows) * cfg.val_frac), replace=False))
    is_val = np.isin(C["q"].to_numpy(), val_q)
    Ctr, Cva = C[~is_val].reset_index(drop=True), C[is_val].reset_index(drop=True)
    Ctr = Ctr.iloc[sample_rows(Ctr, cfg, rng)].reset_index(drop=True)
    Xtr = features_batched(D, Ctr, cfg, tag="train")
    Xva = features_batched(D, Cva, cfg, tag="val")
    booster = train_lgb(Xtr, Ctr["label"].to_numpy(), Xva, Cva["label"].to_numpy(), cfg)
    best_iter = booster.best_iteration or cfg.num_boost_round
    log(f"  LightGBM best iteration {best_iter}")

    prob = booster.predict(Xva.to_numpy(np.float32), num_iteration=best_iter).astype(np.float32)
    booster1 = None
    if cfg.two_stage:
        booster1 = booster
        prob_s1 = prob.copy()
        log("  building stage 2 (out-of-fold stage-1 probabilities)")
        # out-of-fold stage-1 probabilities for ALL candidates of the training entities
        Ctr_all = C[~is_val].reset_index(drop=True)
        tr_q = np.unique(Ctr_all["q"].to_numpy())
        fold_of = np.full(D.n1, -1, np.int8)
        fold_of[tr_q] = rng.integers(0, cfg.oof_folds, len(tr_q))
        f_all, f_smp = fold_of[Ctr_all["q"].to_numpy()], fold_of[Ctr["q"].to_numpy()]
        p1_tr = np.empty(len(Ctr_all), np.float32)
        ytr1 = Ctr["label"].to_numpy()
        for f in range(cfg.oof_folds):
            sel = f_smp != f
            bf = train_lgb(Xtr[sel], ytr1[sel], None, None, cfg, rounds=best_iter)
            idx = np.flatnonzero(f_all == f)
            p1_tr[idx] = features_batched(D, Ctr_all.iloc[idx], cfg, booster=bf, tag=f"oof fold {f}")
            del bf
        C2 = pd.concat([Ctr_all, Cva], ignore_index=True)
        C2 = add_p1_features(C2, np.r_[p1_tr, prob])
        D.anchors_p, D.anchor_pw = compute_anchors(C2, D.n1, cfg.n_anchors, score=C2["p1"].to_numpy(), with_scores=True)
        is_val2 = np.r_[np.zeros(len(Ctr_all), bool), np.ones(len(Cva), bool)]
        Ctr = C2[~is_val2].reset_index(drop=True)
        Cva = C2[is_val2].reset_index(drop=True)
        del C2, Ctr_all
        gc.collect()
        Ctr = Ctr.iloc[sample_rows(Ctr, cfg, rng)].reset_index(drop=True)
        Xtr = features_batched(D, Ctr, cfg, tag="train s2", stage=2)
        Xva = features_batched(D, Cva, cfg, tag="val s2", stage=2)
        booster = train_lgb(Xtr, Ctr["label"].to_numpy(), Xva, Cva["label"].to_numpy(), cfg)
        best_iter = booster.best_iteration or cfg.num_boost_round
        log(f"  stage-2 LightGBM best iteration {best_iter}")
        prob = booster.predict(Xva.to_numpy(np.float32), num_iteration=best_iter).astype(np.float32)
    qi = np.searchsorted(val_q, Cva["q"].to_numpy()).astype(np.int64)
    gt_cnt = np.bincount(np.searchsorted(val_q, D.gt_q[np.isin(D.gt_q, val_q)]), minlength=len(val_q))
    y = Cva["label"].to_numpy().astype(bool)
    tb = target_best(Cva["t"].to_numpy(), prob)
    one = {"on": True, "off": False}.get(cfg.one_to_one, st["targets_with_2plus_s1"] < 0.005)

    def score(thr, oto=one):
        keep = (prob >= thr) & (tb if oto else True)
        return macro_f05(len(val_q), qi[keep], y[keep], gt_cnt)

    s1_best = None
    if cfg.two_stage:
        tb1 = target_best(Cva["t"].to_numpy(), prob_s1)
        s1_best = max(macro_f05(len(val_q), qi[(prob_s1 >= x) & (tb1 if one else True)],
                                y[(prob_s1 >= x) & (tb1 if one else True)], gt_cnt) for x in cfg.threshold_grid)
        log(f"  stage-1 best val F0.5 (for comparison): {s1_best:.4f}")
    curve = pd.DataFrame({"threshold": cfg.threshold_grid, "val_f05": [score(x) for x in cfg.threshold_grid]})
    plateau = curve.loc[curve["val_f05"] >= curve["val_f05"].max() - 5e-4, "threshold"]
    thr = cfg.fixed_threshold if cfg.fixed_threshold is not None else float(plateau.iloc[len(plateau) // 2])
    report = {"k_per_source": int(k), "one_to_one": bool(one), "threshold": thr, "best_iter": int(best_iter),
              "val_entities": int(len(val_q)), "val_f05": score(thr), "val_f05_at_0.75": score(0.75),
              "val_f05_at_0.80": score(0.80), "val_f05_no_one_to_one": score(thr, False),
              "val_f05_blocking_ceiling": macro_f05(len(val_q), qi[y], y[y], gt_cnt),
              "val_f05_predict_nothing": macro_f05(len(val_q), qi[:0], y[:0], gt_cnt),
              "val_f05_stage1_best": s1_best, "profile": st}
    for kk, v in report.items():
        if kk != "profile":
            log(f"  {kk}: {v:.4f}" if isinstance(v, float) else f"  {kk}: {v}")

    # final model: train + sampled validation rows, validated number of rounds (+10 %)
    vi = sample_rows(Cva.assign(label=y.astype(np.int8)), cfg, rng)
    Xall = pd.concat([Xtr, Xva.iloc[vi]], ignore_index=True)
    yall = np.r_[Ctr["label"].to_numpy(), y[vi].astype(np.int8)]
    final = train_lgb(Xall, yall, None, None, cfg, rounds=max(50, int(best_iter * 1.1)))
    imp = pd.Series(final.feature_importance("gain"), index=final.feature_name()).sort_values(ascending=False)
    return {"booster": final, "booster1": booster1, "k": int(k), "threshold": float(thr), "one_to_one": bool(one),
            "curve": curve, "report": report, "importance": imp,
            "val": Cva.assign(prob=prob, label=y.astype(np.int8)),
            "val_gt_cnt": np.c_[val_q, gt_cnt]}


# %% [markdown]
# ## Step 6 — Test inference & output files
# All test S1 rows are queried with the chosen K; features + prediction run in batches so only
# the probabilities are kept. Lists are assembled with Arrow and written without quoting.
# `candidate_pairs.tsv` is exactly the set the model scored; matches are a subset of it.

# %%
def write_lists(path: Path, D: Split, q: np.ndarray, t: np.ndarray, order_score: np.ndarray, colname: str):
    order = np.lexsort((-order_score, q))
    offs = np.concatenate([[0], np.cumsum(np.bincount(q, minlength=D.n1))]).astype(np.int64)
    vals = D.ids.slice(D.n1).take(pa.array(t[order]))
    lists = pc.binary_join(pa.LargeListArray.from_arrays(pa.array(offs), vals), pa.scalar(",", pa.large_string()))
    tab = pa.table({"source1_entity_id": D.ids.slice(0, D.n1), colname: lists})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:   # header written by hand: pyarrow would quote it
        f.write(f"source1_entity_id\t{colname}\n".encode())
        pacsv.write_csv(tab, f, write_options=pacsv.WriteOptions(delimiter="\t", quoting_style="none",
                                                                include_header=False))


def validate_outputs(match_path: Path, cand_path: Path, D: Split) -> bool:
    """Vectorised copy of the submission rules (still run the official validator)."""
    s1 = D.ids.slice(0, D.n1)
    tids = D.ids.slice(D.n1)
    issues, keys = [], {}
    for p, c in ((match_path, "matched_entity_ids"), (cand_path, "candidate_entity_ids")):
        with open(p, encoding="utf-8") as f:
            hdr = f.readline().rstrip("\n")
        if hdr != f"source1_entity_id\t{c}":
            issues.append(f"{p.name}: header {hdr!r}")
        tab = read_tsv(p)
        sid = col(tab, "source1_entity_id")
        if tab.num_rows != D.n1 or not pc.all(pc.equal(sid, s1)).as_py():
            issues.append(f"{p.name}: rows do not match test S1 ids one-to-one")
        L = pc.split_pattern(col(tab, c), ",")
        par = pc.list_parent_indices(L).to_numpy()
        flat = pc.list_flatten(L)
        ne = pc.not_equal(flat, "").to_numpy(zero_copy_only=False)
        pos = pc.index_in(flat, value_set=tids).to_numpy(zero_copy_only=False)
        pos = np.where(ne, pos, -1).astype(float)
        bad = ne & np.isnan(pos)
        if bad.any():
            issues.append(f"{p.name}: {int(bad.sum())} ids not in test S2/S3")
        k = par[ne & ~bad].astype(np.int64) * D.nT + pos[ne & ~bad].astype(np.int64)
        if len(k) != len(np.unique(k)):
            issues.append(f"{p.name}: duplicate ids inside a list")
        keys[c] = k
    if not np.isin(keys["matched_entity_ids"], keys["candidate_entity_ids"]).all():
        issues.append("matched ids not contained in candidate_pairs")
    log("VALIDATION PASS" if not issues else "VALIDATION FAILED: " + "; ".join(issues))
    return not issues


def predict_test(res: dict, cfg: Config) -> dict:
    D = prepare_split(cfg, "test")
    C = generate_candidates(D, np.arange(D.n1, dtype=np.int32), res["k"], cfg)
    if cfg.target_features:
        C = add_target_features(C)
    if cfg.cross_features:
        D.anchors = compute_anchors(C, D.n1, cfg.n_anchors)
    if cfg.two_stage:
        p1 = features_batched(D, C, cfg, booster=res["booster1"], tag="test s1")
        C = add_p1_features(C, p1)
        del p1
        D.anchors_p, D.anchor_pw = compute_anchors(C, D.n1, cfg.n_anchors, score=C["p1"].to_numpy(), with_scores=True)
        prob = features_batched(D, C, cfg, booster=res["booster"], tag="test s2", stage=2)
    else:
        prob = features_batched(D, C, cfg, booster=res["booster"], tag="test")
    q, t = C["q"].to_numpy(), C["t"].to_numpy()
    cache = Path(cfg.model_dir) / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(cache / "test_scored.npz", q=q, t=t, prob=prob, block_score=C["block_score"].to_numpy(),
             src=C["src"].to_numpy())
    log(f"  cached test candidates + probabilities to {cache}")
    keep = prob >= res["threshold"]
    if res["one_to_one"]:
        keep &= target_best(t, prob)
    out = Path(cfg.output_dir)
    write_lists(out / "candidate_pairs.tsv", D, q, t, C["block_score"].to_numpy(), "candidate_entity_ids")
    write_lists(out / "matching_results.tsv", D, q[keep], t[keep], prob[keep], "matched_entity_ids")
    has = np.bincount(q[keep], minlength=D.n1) > 0
    log(f"wrote outputs: {D.n1:,} S1 rows, {int(keep.sum()):,} matches, {int((~has).sum()):,} predicted singletons")
    lab = np.where(D.country[:D.n1] >= 0, D.country_labels[np.maximum(D.country[:D.n1], 0)], "")
    log("share with >=1 match by country: " + ", ".join(
        f"{c or '<empty>'}={has[lab == c].mean():.3f}" for c in pd.unique(lab)))
    validate_outputs(out / "matching_results.tsv", out / "candidate_pairs.tsv", D)
    return {"split": D, "cand": C, "prob": prob, "keep": keep}


def run(cfg: Config) -> dict:
    log(f"threads={cfg.n_threads}  sparse_dot_topn={'yes' if HAVE_SDT else 'no (scipy fallback)'}")
    train = prepare_split(cfg, "train")
    res = train_and_tune(train, cfg)
    del train
    gc.collect()
    md = Path(cfg.model_dir)
    md.mkdir(parents=True, exist_ok=True)
    res["booster"].save_model(str(md / "lgbm_final.txt"))
    if res.get("booster1") is not None:
        res["booster1"].save_model(str(md / "lgbm_stage1.txt"))
    (md / "run_report.json").write_text(json.dumps({"config": asdict(cfg), "validation": res["report"]}, indent=2, default=str))
    res["curve"].to_csv(md / "threshold_curve.tsv", sep="\t", index=False)
    res["importance"].rename("gain").to_csv(md / "feature_importance.tsv", sep="\t")
    (md / "cache").mkdir(exist_ok=True)
    res["val"][["q", "t", "src", "block_score", "prob", "label"]].to_parquet(md / "cache" / "val_scored.parquet")
    np.save(md / "cache" / "val_gt_cnt.npy", res["val_gt_cnt"])
    if cfg.skip_test:
        log("skip_test: stopping after validation")
        return {"train_result": res}
    pred = predict_test(res, cfg)
    log("done")
    return {"train_result": res, "pred": pred}
