#!/usr/bin/env python3
import argparse
import csv
import pickle
import sys
import types
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import training as T


def find_checkpoint(model_dir: Path, epoch):
    if epoch == "best":
        ckpts = sorted(model_dir.glob("epoch_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {model_dir}")
        return ckpts[-1]
    ckpt = model_dir / f"epoch_{epoch}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} does not exist")
    return ckpt


def load_run_params(model_dir: Path):
    with open(model_dir / "params.pkl", "rb") as f:
        params = pickle.load(f)
    keys = ["dataset_folder", "cls", "batch_size", "epochs", "hidden",
            "heads", "layers", "r_bias", "tokenization", "logic"]
    return dict(zip(keys, params))


def load_vocab(model_dir: Path):
    with open(model_dir / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    return vocab


def load_predicate_info(project_root: Path, dataset_folder: str):
    dpath = project_root / "dataset" / dataset_folder / "params.pkl"
    with open(dpath, "rb") as f:
        params = pickle.load(f)
    if isinstance(params, dict) and "predicates" in params:
        return params["predicates"]
    raise ValueError(f"{dpath} does not look like an FOL corpus params file")


def _ensure_classes_fol():
    try:
        import classes_fol
        return
    except ModuleNotFoundError:
        pass
    mod = types.ModuleType("classes_fol")

    class P:
        def __init__(self, name=None, arity=None):
            self._name = name
            self.arity = arity

    mod.P = P
    sys.modules["classes_fol"] = mod


def load_interpretation(project_root: Path, dataset_folder: str):
    _ensure_classes_fol()
    apath = project_root / "dataset" / dataset_folder / "act_world.pkl"
    with open(apath, "rb") as f:
        act_world = pickle.load(f)
    extension = {}
    for pred_obj, tuples in act_world:
        extension[pred_obj._name] = set(tuples)
    return extension


def same_arity_pairs(pred_names, pred_arities, pred_vecs, extension):
    """
    Pairwise tuple overlap (raw intersection count) and cosine similarity,
    restricted to pairs of predicates sharing the same arity. Cross-arity
    pairs are not computed at all: they aren't part of this design (see
    module docstring), so there's no reason to spend time/plot space on
    them here.
    """
    rows_by_arity = {}
    name_to_idx = {n: i for i, n in enumerate(pred_names)}
    for i, j in combinations(range(len(pred_names)), 2):
        n1, n2 = pred_names[i], pred_names[j]
        a1, a2 = pred_arities[n1], pred_arities[n2]
        if a1 != a2:
            continue

        e1, e2 = extension.get(n1, set()), extension.get(n2, set())
        overlap = len(e1 & e2)  # raw tuple-intersection count ("tuple overlap")

        v1, v2 = pred_vecs[name_to_idx[n1]], pred_vecs[name_to_idx[n2]]
        cos_sim = float(np.dot(v1, v2) /
                         (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))

        rows_by_arity.setdefault(a1, []).append((n1, n2, overlap, cos_sim))
    return rows_by_arity


def _rankdata(a: np.ndarray):
    a = np.asarray(a)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    sorted_a = a[sorter]
    ranks = inv.astype(float) + 1
    i = 0
    while i < len(sorted_a):
        j = i
        while j + 1 < len(sorted_a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            tied_positions = sorter[i:j + 1]
            avg_rank = ranks[tied_positions].mean()
            ranks[tied_positions] = avg_rank
        i = j + 1
    return ranks


def spearman(xs, ys):
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    rx, ry = _rankdata(xs), _rankdata(ys)
    return float(np.corrcoef(rx, ry)[0, 1])


def build_model(run_params, vocab_size, max_len, pad_id):
    T.hidden = run_params["hidden"]
    T.heads = run_params["heads"]
    T.layers = run_params["layers"]
    T.vocab_size = vocab_size
    T.max_len = max_len
    T.pad_id = pad_id
    model = T.EncoderTransformer()
    return model


def reduce_to_2d(vectors: np.ndarray):
    from sklearn.manifold import TSNE
    perplexity = max(2, min(30, vectors.shape[0] - 1))
    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=0).fit_transform(vectors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--epoch", default="best")
    ap.add_argument("--output", type=str, default="predicate_embeddings.png")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent
    model_dir = project_root / "model" / args.model_dir

    run_params = load_run_params(model_dir)

    vocab = load_vocab(model_dir)
    tok_to_id = {tok: i for i, tok in enumerate(vocab)}
    vocab_size = len(vocab)
    pad_id = tok_to_id["[PAD]"]

    predicates = load_predicate_info(project_root, run_params["dataset_folder"])
    pred_names = [name for name, arity in predicates]
    pred_arities = {name: arity for name, arity in predicates}

    missing = [n for n in pred_names if n not in tok_to_id]
    if missing:
        raise ValueError(f"predicate symbols not found in vocab: {missing}")

    ckpt = find_checkpoint(model_dir, args.epoch)
    state_dict = torch.load(ckpt, map_location="cpu")
    max_len_for_init = state_dict["pos_emb.weight"].shape[0]

    model = build_model(run_params, vocab_size, max_len_for_init, pad_id)
    model.load_state_dict(state_dict)
    model.eval()

    tok_emb = model.tok_emb.weight.detach().cpu().numpy()
    pred_ids = [tok_to_id[n] for n in pred_names]
    pred_vecs = tok_emb[pred_ids]

    # --- unchanged: general t-SNE layout of predicate embeddings ---
    # Kept as-is; it's a different visualization from the overlap
    # experiment below and wasn't part of what we're changing here.
    coords = reduce_to_2d(pred_vecs)
    arities = sorted(set(pred_arities.values()))
    cmap = plt.get_cmap("tab10")
    arity_color = {a: cmap(i % 10) for i, a in enumerate(arities)}

    plt.figure(figsize=(8, 8))
    for (x, y), name in zip(coords, pred_names):
        color = arity_color[pred_arities[name]]
        plt.scatter(x, y, color=color, s=80)
        plt.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5))
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=arity_color[a], markersize=10,
                           label=f"arity {a}") for a in arities]
    plt.legend(handles=handles, title="predicate arity")
    epoch_label = ckpt.stem.split("_")[1]
    plt.title(f"Static predicate embeddings\n{model_dir.name} (epoch {epoch_label})")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"saved t-SNE plot to {args.output}")

    # --- the actual experiment: overlap vs. embedding similarity, per arity ---
    extension = load_interpretation(project_root, run_params["dataset_folder"])
    rows_by_arity = same_arity_pairs(pred_names, pred_arities, pred_vecs, extension)

    all_csv_rows = []
    print(f"\n=== per-arity tuple-overlap vs. embedding-similarity ===")
    for a in sorted(rows_by_arity.keys()):
        group = rows_by_arity[a]
        n1s = [r[0] for r in group]
        n2s = [r[1] for r in group]
        overlaps = [r[2] for r in group]
        cos_sims = [r[3] for r in group]

        print(f"\narity {a}: {len(group)} pairs, "
              f"extension size fixed by construction for this arity")
        if len(group) < 3:
            print("  too few pairs to compute a meaningful correlation")
            rho = float("nan")
        else:
            rho = spearman(overlaps, cos_sims)
            print(f"  Spearman correlation (tuple overlap vs. cosine similarity): {rho:.4f}")

        for n1, n2, ov, cs in zip(n1s, n2s, overlaps, cos_sims):
            all_csv_rows.append((n1, n2, a, ov, cs))

        # one figure per arity, as agreed -- pooling arities together would
        # mix groups whose extension size (and therefore overlap scale)
        # differs by construction across arities.
        plt.figure(figsize=(6, 5))
        plt.scatter(overlaps, cos_sims, alpha=0.8, color="tab:blue")
        for n1, n2, ov, cs in zip(n1s, n2s, overlaps, cos_sims):
            plt.annotate(f"{n1}{n2}", (ov, cs), fontsize=6,
                         textcoords="offset points", xytext=(3, 3), alpha=0.6)
        plt.xlabel("tuple overlap (intersection count)")
        plt.ylabel("cosine similarity between static embeddings")
        plt.title(f"Arity {a} predicates: overlap vs. embedding similarity\n"
                  f"{model_dir.name} (epoch {epoch_label}), Spearman ρ={rho:.3f}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        arity_plot_path = f"{args.model_dir}_arity{a}_preds_plot.png"
        plt.savefig(arity_plot_path, dpi=150)
        print(f"  saved plot to {arity_plot_path}")

    # single CSV with all same-arity pairs, arity column included so the
    # per-arity groups can still be recovered/re-analyzed later
    pairs_csv_path = f"pairs_{args.model_dir}.csv"
    with open(pairs_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pred_1", "pred_2", "arity", "tuple_overlap", "cosine_similarity"])
        w.writerows(all_csv_rows)
    print(f"\nsaved pairwise table (same-arity pairs only) to {pairs_csv_path}")


if __name__ == "__main__":
    main()