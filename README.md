# Face Generation Challenge — Submission Reproduction Package

Regenerates our challenge submissions from a single checkpoint with plain
`python` (no cluster/qsub). **One environment, one script** — pick which
per-metric image set to build with a `--config` option. The model weight is
provided separately (see **Checkpoint**); all the code is in this repo.

## Scoring & what this produces

The challenge ranks every submission **per metric** (FID, IS, KID, TopPR) and a
participant's final standing is the **average of those four ranks**. No single
generation setting tops all four, so we keep the setting that wins each metric;
this package can reproduce each one — pick with `--config`:

| `--config` | best metric(s) | how it is made | extra input |
|---|---|---|---|
| `psi10` *(default)* | **FID, KID** | psi=1.0, seeds 0–999, no selection | none |
| `psi20` | **IS** | psi=2.0, seeds 0–999, no selection | none |
| `prune085` | **TopPR** | psi=0.85, 5000-pool → support-prune to 1000 | feature cache (bundled) |

All three come from `checkpoint/fdloss-320.pkl` (StyleGAN2-ADA fine-tuned on
CelebV-HQ, then post-trained with FD-Loss). `psi10` is our balanced set.

Local scores (1000 generated vs CelebV-HQ, our eval pipeline):

| config | FID ↓ | KID ↓ | IS ↑ | TopPR-F1 ↑ |
|---|---|---|---|---|
| `psi10`    | 30.25 | 0.00277 | 4.68 | 0.841 |
| `psi20`    | 112.5 | 0.0802  | 6.32 | 0.230 |
| `prune085` | 43.31 | 0.0159  | 3.36 | 0.935 |

(`psi10` server score: FID 28.97 · IS 4.66 · KID 0.0017 · TopPR 0.837.)
Each config wins its target metric but gives up the others — that trade-off is
exactly why the per-metric mapping above exists.

> **Note (relative to the report).** Our submitted report describes all three
> configs: `psi10` is the submission, while `psi20` (IS-max) and `prune085`
> (TopPR-max) appear in the metric trade-off analysis (Table 1). The report
> states that the shared code reproduces only `psi10`; this package has since
> been extended so that `psi20` and `prune085` are reproducible here as well.

## Folder contents

```
submission_repo/
├── README.md           ← this file
├── reproduce.py        ← build the submission folder for a chosen --config
├── requirements.txt    ← all deps (generation + selection), one env
├── checkpoint/         ← place fdloss-320.pkl here (bundled separately; see below)
├── ref/                ← place celebvhq256_cleanfeats.npz here (prune085 only)
└── stylegan2_ada/      ← minimal NVlabs StyleGAN2-ADA inference code
    ├── generate.py  legacy.py
    ├── dnnlib/  torch_utils/  training/
```

`checkpoint/fdloss-320.pkl` (~282 MB) and `ref/celebvhq256_cleanfeats.npz`
(~584 MB, prune085 only) are large derived files: they are **provided with the
submission bundle (email)**, not in the git repo (GitHub's 100 MB file limit).

## Environment

One environment covers everything. The generator compiles custom CUDA ops at
first run, so it needs CUDA 11.3 + gcc 8; the `prune085` selection adds
`clean-fid` + `scikit-learn` (already in `requirements.txt`). Verified:
**Python 3.9, torch 1.12.1+cu113, RTX 3090**.

```bash
conda create -n repro python=3.9 -y && conda activate repro
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## Checkpoint & feature cache

Two large files ship with the email bundle (not in the git repo):

- `checkpoint/fdloss-320.pkl` (~282 MB) — the generator weight. Needed by every
  config.
- `ref/celebvhq256_cleanfeats.npz` (~584 MB) — clean-fid Inception features of
  the real CelebV-HQ set. Needed **only by `prune085`**, which selects the 1000
  generated images closest to this real-data support. It is the exact (float32)
  cache used for the submission.

Place each at the path shown above before running. `prune085` then needs no
external dataset; pass `--real-dir <celebvhq_256>` only if you want to rebuild
the cache from the public images instead.

## Reproduce

Each run writes a folder `out/submission_<config>/` of 1000 PNGs named
`0000.png`–`0999.png` (one folder per config, so they don't overwrite). A few
minutes on an RTX 3090. Submit the folder directly, or zip it yourself if your
upload channel needs a single file.

```bash
python reproduce.py                    # psi10     (FID/KID, default)
python reproduce.py --config psi20     # psi20     (IS)
python reproduce.py --config prune085  # prune085  (TopPR); uses ref/celebvhq256_cleanfeats.npz
```

With `ref/celebvhq256_cleanfeats.npz` from the bundle in place, `prune085` runs
with no extra flags. To rebuild the cache from scratch instead, point at the
public CelebV-HQ images (256×256 folder); it is extracted once and cached:

```bash
python reproduce.py --config prune085 --real-dir /path/to/celebvhq_256
```

## Determinism

`psi10` / `psi20` are deterministic per seed: NVlabs `generate.py` sets
`z = np.random.RandomState(seed).randn(1, 512)` per image with
`--noise-mode=const`, so seeds 0–999 reproduce exactly (minor GPU/driver float
differences can change PNG bytes without changing the image).

`prune085` generates its 5,000-image pool the same deterministic way, but the
selection step extracts clean-fid Inception features and runs k-means, which
depend on the environment (torch / clean-fid / scikit-learn versions, GPU vs
CPU). Across environments the *exact* 1,000 selected images can differ
substantially — selecting under a different torch than the original, we measured
~38% image overlap — yet the **metric reproduces closely**: that re-run scored
TopPR-F1 0.912 vs the original 0.935, still far above the untruncated set
(0.841). So `prune085` reproduces the TopPR *result*, not a bit-identical image
set.
