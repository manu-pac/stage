"""
        --true-reps reps/my_run/epoch_10__true_dev__mean.pkl \
        --false-reps reps/my_run/epoch_10__false_dev__mean.pkl \
        --output my_run
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def load_reps(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--true-reps", type=str, required=True,
                         help="Path to a reps pickle (from get_representations.py) for the true set.")
    parser.add_argument("--false-reps", type=str, required=True,
                         help="Path to a reps pickle (from get_representations.py) for the false set. "
                              "Must be the same rep type (mean/cls) as --true-reps.")
    parser.add_argument("--output", type=str, required=True,
                         help="Folder name to save the plot into. Always saved under plots/<output>/.")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n-sample", type=int, default=None,
                         help="Subsample this many points total (combined) before running t-SNE. "
                              "Default: use all points.")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    true_data = load_reps(args.true_reps)
    false_data = load_reps(args.false_reps)

    rep_name = true_data["type"]

    true_emb = true_data["reps"]
    false_emb = false_data["reps"]

    embeddings = np.concatenate([true_emb, false_emb], axis=0)
    labels = np.array(["true"] * len(true_emb) + ["false"] * len(false_emb))

    # subsample for speed (optional)
    if args.n_sample is not None and args.n_sample < len(embeddings):
        idx = np.random.choice(len(embeddings), size=args.n_sample, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    # perplexity must be < number of samples
    eff_perplexity = min(args.perplexity, len(embeddings) - 1)
    if eff_perplexity != args.perplexity:
        print(f"reducing perplexity {args.perplexity} -> {eff_perplexity} "
              f"(only {len(embeddings)} points)")

    tsne = TSNE(n_components=2, perplexity=eff_perplexity, random_state=args.random_state,
                init="pca", learning_rate="auto")
    coords = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    for label in ("true", "false"):
        mask = labels == label
        plt.scatter(coords[mask, 0], coords[mask, 1], label=label, alpha=0.6, s=15)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.title(f"t-SNE of {rep_name} representations (true vs false)")
    plt.legend()

    project_root = Path(__file__).resolve().parent
    out_dir = project_root / "plots" / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    true_stem = Path(args.true_reps).stem
    false_stem = Path(args.false_reps).stem
    out_path = out_dir / f"{true_stem}__{false_stem}_tsne.png"

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved plot to {out_path}  ({len(embeddings)} points, perplexity={eff_perplexity})")


if __name__ == "__main__":
    main()