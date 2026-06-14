#!/usr/bin/env python3
"""Reproduce a per-metric submission from the fdloss-320 checkpoint.

One script, one environment. The challenge ranks every submission *per metric*
(FID, IS, KID, TopPR) and a participant's final standing is the average of those
four ranks. No single setting tops all four, so we keep the setting that wins
each one; pick which to reproduce with --config (default: psi10):

  psi10    (default)  FID + KID winner : psi=1.0,  seeds 0-999,  no selection
  psi20               IS winner        : psi=2.0,  seeds 0-999,  no selection
  prune085            TopPR winner     : psi=0.85, 5000-pool -> support-prune 1000

prune085 additionally keeps the 1000 samples closest to the real-data support,
so it needs the real CelebV-HQ images once (via --real-dir) to build a feature
cache; psi10 / psi20 need nothing extra. All deps are in requirements.txt.

Each run writes out/submission_<config>/ as a folder of 1000 PNGs named
0000.png-0999.png (no zip).

Examples:
  python reproduce.py                                    # psi10
  python reproduce.py --config psi20
  python reproduce.py --config prune085 --real-dir /path/to/celebvhq_256
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoint" / "fdloss-320.pkl"
GEN = ROOT / "stylegan2_ada" / "generate.py"
DEFAULT_REF_CACHE = ROOT / "ref" / "celebvhq256_cleanfeats.npz"
IMG_EXT = (".png", ".jpg", ".jpeg")

# config -> generation + optional selection recipe (mirrors our sweep exactly).
CONFIGS = {
    "psi10": {"psi": 1.0, "n_gen": 1000, "metric": "FID + KID", "select": None},
    "psi20": {"psi": 2.0, "n_gen": 1000, "metric": "IS", "select": None},
    "prune085": {"psi": 0.85, "n_gen": 5000, "metric": "TopPR",
                 "select": {"n": 1000, "keep_frac": 0.6, "knn_k": 5,
                            "proj_dim": 64, "seed": 0}},
}


# ---------------------------------------------------------------- generation
def generate(psi, seeds, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(GEN), f"--network={CKPT}", f"--seeds={seeds}",
           f"--trunc={psi}", "--noise-mode=const", f"--outdir={outdir}"]
    print(f"[gen] psi={psi} seeds={seeds} -> {outdir.name}")
    subprocess.run(cmd, check=True)


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir()
                  if p.suffix.lower() in IMG_EXT)


# -------------------------------------------------- prune085 (TopPR) selection
# Faithful port of our sweep's 'support' selection: drop generated samples that
# fall outside the real-data support (k-th NN distance to real features in an
# L2-normalised, random-projected clean-fid Inception space), then MiniBatchKMeans
# the survivors down to n to preserve diversity (recall). Uses clean-fid +
# scikit-learn, imported lazily so psi10 / psi20 never touch them.
def extract_features(folder, device, num_workers):
    from cleanfid import fid
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda unavailable -> cpu")
        device = "cpu"
    dev = torch.device(device)
    model = fid.build_feature_extractor("clean", dev)
    f = fid.get_folder_features(str(folder), model, num_workers=num_workers,
                                device=dev, mode="clean", verbose=False)
    return np.asarray(f, dtype=np.float64)


def load_or_build_ref(ref_cache, real_dir, device, num_workers):
    rp = Path(ref_cache)
    if rp.exists():
        print(f"[ref] using cache {rp}")
        return np.load(rp)["feats"].astype(np.float32)
    if not real_dir:
        raise SystemExit(
            f"[ERR] prune085 needs the real-feature cache. Place the provided "
            f"celebvhq256_cleanfeats.npz at {rp} (default), or pass --ref-cache "
            f"<file.npz>; or pass --real-dir <celebvhq_256> to rebuild it. See README.")
    print(f"[ref] building feature cache from {real_dir} ...")
    feats = extract_features(real_dir, device, num_workers).astype(np.float32)
    rp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(rp, feats=feats)
    print(f"[ref] {feats.shape} real features -> {rp}")
    return feats


def _kmeans_pick(feats, n, seed):
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=n, random_state=seed, batch_size=1024,
                         n_init=3, max_iter=100).fit(feats)
    picked, order = set(), []
    for c in km.cluster_centers_:                 # nearest pool sample per centre
        d = np.linalg.norm(feats - c, axis=1)
        for idx in np.argsort(d):
            ii = int(idx)
            if ii not in picked:
                picked.add(ii)
                order.append(ii)
                break
    if len(order) < n:                            # dedup left us short -> fill
        order.extend(i for i in range(feats.shape[0]) if i not in picked)
    return order[:n]


def _knn_kth_dist(query, ref, k):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute",
                          metric="euclidean").fit(ref)
    dist, _ = nn.kneighbors(query)
    return dist[:, -1]


def select_prune(pool_dir, ref_feats, p, device, num_workers):
    files = list_images(pool_dir)
    print(f"[select] extracting clean-fid features for {len(files)} pool images ...")
    feats = extract_features(pool_dir, device, num_workers)
    assert len(files) == feats.shape[0], "feature/file count mismatch"
    P, D = feats.shape
    f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    r = ref_feats / (np.linalg.norm(ref_feats, axis=1, keepdims=True) + 1e-8)
    rng = np.random.RandomState(p["seed"])        # JL random projection (speed)
    proj = (rng.randn(D, p["proj_dim"]) / np.sqrt(p["proj_dim"])).astype(np.float32)
    d = _knn_kth_dist(f.astype(np.float32) @ proj, r.astype(np.float32) @ proj,
                      p["knn_k"])
    keep_m = max(p["n"], int(P * p["keep_frac"]))
    survivors = np.argsort(d)[:keep_m]            # closest to real support
    print(f"[select] keep {keep_m}/{P} in-support -> k-means {p['n']}")
    local = _kmeans_pick(feats[survivors], p["n"], p["seed"])
    idxs = sorted(int(survivors[i]) for i in local)
    return [files[i] for i in idxs]


# ------------------------------------------------------------------- output
def write_submission(img_paths, dst_dir):
    """Write the 1000 chosen images to a folder as 0000.png..0999.png."""
    assert len(img_paths) == 1000, f"{dst_dir.name}: {len(img_paths)} != 1000"
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True)
    for i, pth in enumerate(img_paths):
        shutil.copy2(pth, dst_dir / f"{i:04d}.png")
    print(f"[out] {dst_dir}  (1000 PNGs: 0000.png..0999.png)")


def main():
    ap = argparse.ArgumentParser(description="Reproduce a per-metric submission.")
    ap.add_argument("--config", choices=list(CONFIGS), default="psi10",
                    help="which per-metric image set to build (default: psi10)")
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--real-dir", default=None,
                    help="prune085 only: real CelebV-HQ 256 folder (cached once)")
    ap.add_argument("--ref-cache", default=str(DEFAULT_REF_CACHE),
                    help="prune085 only: real-feature cache (.npz)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--skip-gen", action="store_true",
                    help="reuse images already in <out>/_gen_<config>")
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    gen_dir = out / f"_gen_{args.config}"

    if not args.skip_gen:
        generate(cfg["psi"], f"0-{cfg['n_gen'] - 1}", gen_dir)

    pool = list_images(gen_dir)
    if len(pool) < cfg["n_gen"]:
        raise SystemExit(f"[ERR] pool {len(pool)} < {cfg['n_gen']} in {gen_dir}")

    if cfg["select"] is None:                     # psi10 / psi20: raw seed order
        chosen = pool[:1000]
    else:                                         # prune085: support-prune
        ref = load_or_build_ref(args.ref_cache, args.real_dir,
                                args.device, args.num_workers)
        chosen = select_prune(gen_dir, ref, cfg["select"],
                              args.device, args.num_workers)

    sub_dir = out / f"submission_{args.config}"
    write_submission(chosen, sub_dir)
    print(f"\n[done] config={args.config} (target: {cfg['metric']}) -> {sub_dir}/")


if __name__ == "__main__":
    main()
