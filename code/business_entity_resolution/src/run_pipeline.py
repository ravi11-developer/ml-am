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
    a = p.parse_args()
    cfg = Config(data_dir=a.data_dir, output_dir=a.output_dir, model_dir=a.model_dir,
                 train_query_sample=a.train_queries or None, k_per_source=a.k, k_max_per_source=a.k_max,
                 max_posting=a.max_posting, rerank_pool=a.pool, fast_features=a.fast_features, fixed_threshold=a.threshold,
                 one_to_one=a.one_to_one, seed=a.seed)
    if a.threads:
        cfg.n_threads = a.threads
    run(cfg)
