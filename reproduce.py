#!/usr/bin/env python3
"""Reproduce our final submission from the fdloss-320 checkpoint.

The challenge ranks by the average of a submission's four per-metric ranks, so we
submit a single balanced image set: truncation psi=1.0, no selection, seeds 0-999.
Self-contained: uses only files inside this folder.

Run in an environment with torch 1.12.x + the StyleGAN2-ADA custom CUDA ops
(see README.md). Example:
  python reproduce.py --out out
Produces out/submission.zip (1000 PNGs named 0000.png-0999.png at the zip root).
"""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoint" / "fdloss-320.pkl"
GEN = ROOT / "stylegan2_ada" / "generate.py"


def generate(psi: float, seeds: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(GEN), f"--network={CKPT}", f"--seeds={seeds}",
           f"--trunc={psi}", "--noise-mode=const", f"--outdir={outdir}"]
    print(f"[gen] psi={psi} seeds={seeds} -> {outdir.name}")
    subprocess.run(cmd, check=True)


def build_zip(img_paths, zip_path: Path):
    assert len(img_paths) == 1000, f"{zip_path.name}: {len(img_paths)} != 1000"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for i, p in enumerate(img_paths):
            zf.write(p, arcname=f"{i:04d}.png")
    print(f"[zip] {zip_path.name}  ({zip_path.stat().st_size/1e6:.1f} MB, 1000 imgs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--skip-gen", action="store_true",
                    help="reuse existing generated images in --out")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    gen_dir = out / "_gen"
    if not args.skip_gen:
        generate(1.0, "0-999", gen_dir)

    # Final submission: raw, sorted seed order -> 0000.png..0999.png
    build_zip(sorted(gen_dir.glob("seed*.png")), out / "submission.zip")

    print("\n[done] submission.zip in", out)
    print("  final submission: psi=1.0, seeds 0-999, no selection "
          "(server FID 28.97 / IS 4.66 / KID 0.0017 / TopPR 0.837)")


if __name__ == "__main__":
    main()
