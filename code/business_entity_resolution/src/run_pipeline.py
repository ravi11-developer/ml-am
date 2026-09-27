"""CLI: python src/run_pipeline.py --data-dir <dataset> --output-dir <output>  (see README for options)"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from er_pipeline import Config, run  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Business Entity Resolution pipeline (large-scale)")
    p.add_argument("--data-dir", default="dataset"); p.add_argument("--output-dir", default="output")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--train-queries", type=int, default=200_000, help="S1 train entities used as queries (0 = all)")
    p.add_argument("--k", type=int, default=None, help="fixed candidates per source (default: auto from recall curve)")
    p.add_argument("--k-max", type=int, default=20)
    p.add_argument("--max-posting", type=int, default=3000, help="lower = faster blocking, less recall")
    p.add_argument("--pool", type=int, default=100, help="retrieval pool per source before full-cosine re-rank")
    p.add_argument("--fast-features", action="store_true", help="skip the 3 slowest string features")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--one-to-one", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k-name-pass", type=int, default=0, help="extra name-only retrieval pass, top-k per source (0 = off)")
    p.add_argument("--k-addr-pass", type=int, default=0, help="extra address-only retrieval pass, top-k per source (0 = off)")
    p.add_argument("--extra-pool", type=int, default=100, help="retrieval pool of the extra passes")
    p.add_argument("--k-rev-pass", type=int, default=0, help="reverse pass: each S2/S3 record retrieves its top-k S1 (0 = off)")
    p.add_argument("--rev-pool", type=int, default=30, help="retrieval pool of the reverse pass")
    p.add_argument("--leaves", type=int, default=127, help="LightGBM num_leaves")
    p.add_argument("--indic-translit", action="store_true", help="romanise Indic-script names/addresses")
    p.add_argument("--name-skeleton", action="store_true", help="add phonetic-skeleton keys to the name block")
    p.add_argument("--name-clean", action="store_true", help="names: leet digits -> letters, drop web tokens")
    p.add_argument("--cross-features", action="store_true", help="similarity to the entity's other top candidates")
    p.add_argument("--skip-test", action="store_true", help="stop after training / validation")
    p.add_argument("--rounds", type=int, default=1500, help="max LightGBM boosting rounds (early stopping on validation)")
    p.add_argument("--lr", type=float, default=0.08, help="LightGBM learning rate")
    p.add_argument("--max-train-rows", type=int, default=3_000_000, help="cap on sampled training pairs")
    p.add_argument("--two-stage", action="store_true", help="stage-2 model on out-of-fold stage-1 probabilities")
    p.add_argument("--house-fuzzy", action="store_true", help="house-number match tolerant to truncation / padding")
    p.add_argument("--addr-numbers", action="store_true", help="compare all address numbers (decoys shift house numbers)")
    p.add_argument("--sib-features", action="store_true", help="count S1 entities sharing the canonical name")
    p.add_argument("--norm-v2", action="store_true", help="acronyms, French street abbreviations, core-name variants")
    p.add_argument("--miss-features", action="store_true", help="rarest name/address token missing from the other record")
    p.add_argument("--target-features", action="store_true",
                   help="block all train S1 (as at test) and add target-side competition features (needs ~50 GB RAM)")
    a = p.parse_args()
    cfg = Config(data_dir=a.data_dir, output_dir=a.output_dir, model_dir=a.model_dir,
                 train_query_sample=a.train_queries or None, k_per_source=a.k, k_max_per_source=a.k_max,
                 max_posting=a.max_posting, rerank_pool=a.pool, fast_features=a.fast_features, fixed_threshold=a.threshold,
                 one_to_one=a.one_to_one, seed=a.seed, k_name_pass=a.k_name_pass, k_addr_pass=a.k_addr_pass,
                 extra_pool=a.extra_pool, k_rev_pass=a.k_rev_pass, rev_pool=a.rev_pool, indic_translit=a.indic_translit, name_skeleton=a.name_skeleton,
                 name_clean=a.name_clean, cross_features=a.cross_features, skip_test=a.skip_test,
                 two_stage=a.two_stage, house_fuzzy=a.house_fuzzy, target_features=a.target_features,
                 addr_numbers=a.addr_numbers, sib_features=a.sib_features, norm_v2=a.norm_v2,
                 miss_features=a.miss_features)
    cfg.num_boost_round, cfg.max_train_rows = a.rounds, a.max_train_rows
    cfg.lgb_params = dict(cfg.lgb_params, learning_rate=a.lr, num_leaves=a.leaves)
    if a.threads:
        cfg.n_threads = a.threads
    run(cfg)
