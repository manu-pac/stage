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


def pairwise_overlap_and_distance(pred_names, pred_vecs, extension, metric="jaccard"):
    rows = []
    for i, j in combinations(range(len(pred_names)), 2):
        n1, n2 = pred_names[i], pred_names[j]
        e1, e2 = extension.get(n1, set()), extension.get(n2, set())
        inter = len(e1 & e2)
        union = len(e1 | e2)
        jaccard = inter / union if union else 0.0
        overlap = jaccard if metric == "jaccard" else inter

        v1, v2 = pred_vecs[i], pred_vecs[j]
        cos_sim = float(np.dot(v1, v2) /
                         (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
        cos_dist = 1.0 - cos_sim
        rows.append((n1, n2, inter, jaccard, overlap, cos_dist))
    return rows


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


def correlation(xs, ys):
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    pearson = float(np.corrcoef(xs, ys)[0, 1])
    rx, ry = _rankdata(xs), _rankdata(ys)
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return pearson, spearman


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
    ap.add_argument("--overlap_metric", choices=["jaccard", "count"], default="jaccard")
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
    print(f"saved plot to {args.output}")

    extension = load_interpretation(project_root, run_params["dataset_folder"])
    rows = pairwise_overlap_and_distance(pred_names, pred_vecs, extension,
                                          metric=args.overlap_metric)
    overlaps = [r[4] for r in rows]
    cos_dists = [r[5] for r in rows]
    pearson, spearman = correlation(overlaps, cos_dists)
    print(f"pairs analyzed: {len(rows)}")
    print(f"overlap metric: {args.overlap_metric}")
    print(f"Pearson correlation (overlap vs cosine distance):  {pearson:.4f}")
    print(f"Spearman correlation (overlap vs cosine distance): {spearman:.4f}")

    pairs_csv_path = f"pairs_{args.model_dir}.csv"
    with open(pairs_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pred_1", "pred_2", "intersection_count", "jaccard",
                    "overlap_used", "cosine_distance"])
        w.writerows(rows)
    print(f"saved pairwise table to {pairs_csv_path}")

    plt.figure(figsize=(7, 6))
    plt.scatter(overlaps, cos_dists, alpha=0.7)
    for n1, n2, _inter, _jac, overlap, cdist in rows:
        plt.annotate(f"{n1}{n2}", (overlap, cdist), fontsize=6,
                     textcoords="offset points", xytext=(3, 3), alpha=0.6)
    xlabel = ("Jaccard overlap of extensions" if args.overlap_metric == "jaccard"
              else "shared individuals/groups (intersection count)")
    plt.xlabel(xlabel)
    plt.ylabel("cosine distance between embeddings")
    plt.title(f"Embedding distance vs. shared interpretation\n"
              f"Pearson r={pearson:.3f}, Spearman ρ={spearman:.3f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    corr_plot_path = f"{args.model_dir}_preds_plot.png"
    plt.savefig(corr_plot_path, dpi=150)
    print(f"saved correlation plot to {corr_plot_path}")


if __name__ == "__main__":
    main()