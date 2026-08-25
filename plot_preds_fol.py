import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import training as T

def find_checkpoint(model_name: Path, epoch):
    if epoch == "best":
        ckpts = sorted(model_name.glob("epoch_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {model_name}")
        return ckpts[-1]
    ckpt = model_name / f"epoch_{epoch}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} does not exist")
    return ckpt

def load_run_params(model_name: Path):
    with open(model_name / "params.pkl", "rb") as f:
        params = pickle.load(f)
    keys = ["dataset_folder", "cls", "batch_size", "epochs", "hidden",
            "heads", "layers", "r_bias", "tokenization", "logic"]
    return dict(zip(keys, params))


def load_vocab(model_name: Path):
    with open(model_name / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    return vocab


def load_predicate_info(project_root: Path, dataset_folder: str):
    dpath = project_root / "dataset" / dataset_folder / "params.pkl"
    with open(dpath, "rb") as f:
        params = pickle.load(f)
    if isinstance(params, dict) and "predicates" in params:
        return params["predicates"]  # list of (name, arity)
    raise ValueError(f"{dpath} does not look like an FOL corpus params file "
                      f"(expected a dict with a 'predicates' key)")


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
    ap.add_argument("--model_name", type=str, required=True,
                    help="path to the training output folder (contains "
                         "epoch_*.pt, params.pkl, vocab.pkl)")
    ap.add_argument("--epoch", default="best",
                    help="epoch number to load, or 'best' for the last "
                         "checkpoint written (no best-epoch marker is saved "
                         "to disk, so 'best' just means most recent)")
    ap.add_argument("--output", type=str, default="predicate_embeddings.png")
    args = ap.parse_args()

    model_name = Path(args.model_name).resolve()
    project_root = Path(__file__).resolve().parent

    run_params = load_run_params(model_name)

    vocab = load_vocab(model_name)
    tok_to_id = {tok: i for i, tok in enumerate(vocab)}
    vocab_size = len(vocab)
    pad_id = tok_to_id["[PAD]"]

    predicates = load_predicate_info(project_root, run_params["dataset_folder"])
    pred_names = [name for name, arity in predicates]
    pred_arities = {name: arity for name, arity in predicates}

    missing = [n for n in pred_names if n not in tok_to_id]
    if missing:
        raise ValueError(f"predicate symbols not found in vocab: {missing}")

    ckpt = find_checkpoint(model_name, args.epoch)
    state_dict = torch.load(ckpt, map_location="cpu")
    # read max_len straight off the saved positional embedding table so we
    # don't need to recompute it from the corpus
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
    plt.title(f"Static predicate embeddings\n{model_name.name} (epoch {epoch_label})")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"saved plot to {args.output}")


if __name__ == "__main__":
    main()