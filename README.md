# Face Generation Challenge — Submission Reproduction Package

Package that regenerates our **final submission** from a **single checkpoint**
using plain `python` (no cluster/qsub). The model weight is provided separately
(see **Checkpoint** below); all the code is in this repo.

## What this produces

The challenge ranks each submission by the **average of its four per-metric ranks**
(FID, IS, KID, TopPR), so a competitive entry must be strong on all four at once.
Our submission is therefore a single **balanced** image set, generated from
`checkpoint/fdloss-320.pkl` (StyleGAN2-ADA fine-tuned on CelebV-HQ, then
post-trained with FD-Loss) at truncation `psi=1.0` with no selection:

| zip | `psi` | selection | seeds | server score |
|---|---|---|---|---|
| `submission.zip` | 1.0 | none | 0–999 | FID 28.97 · IS 4.66 · KID 0.0017 · TopPR 0.837 → **2nd overall** |

## Folder contents

```
submission_repo/
├── README.md            ← this file
├── reproduce.py         ← generate submission.zip
├── requirements.txt     ← python deps + CUDA-toolchain note
├── checkpoint/          ← place fdloss-320.pkl here (not in repo; see Checkpoint)
└── stylegan2_ada/       ← minimal NVlabs StyleGAN2-ADA inference code
    ├── generate.py  legacy.py
    ├── dnnlib/  torch_utils/  training/
```

## Environment

The generator compiles custom CUDA ops at first run, so it needs CUDA 11.3 + gcc 8.
Verified config: **Python 3.9, torch 1.12.1+cu113, RTX 3090**.

```bash
conda create -n repro python=3.9 -y && conda activate repro
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## Checkpoint

The generator weight `fdloss-320.pkl` (~282 MB) is **submitted separately** and is
not stored in this repository (GitHub's 100 MB file limit; challenge weights are
submitted out-of-band). Before running, place it at `checkpoint/fdloss-320.pkl`.

## Reproduce

```bash
python reproduce.py --out out
```
This generates `out/submission.zip` (1000 PNGs named `0000.png`–`0999.png` at the
zip root). A few minutes on an RTX 3090.

## Determinism

Generation is pure per-seed: NVlabs `generate.py` sets
`z = np.random.RandomState(seed).randn(1, 512)` per image with `--noise-mode=const`,
so seeds 0–999 reproduce deterministically and do not depend on cuDNN flags. Minor
GPU/driver float differences can change PNG bytes without changing the image.
