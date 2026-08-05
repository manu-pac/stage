"""
        --model \
        --epoch \
        --output 

visualizes the model's non-contextual token embedding table
"""

import argparse
import pickle
import string
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE


def build_vocab(number_pl, use_cls, tokenization="char"):
    # mirrors reps.py / training.py's vocab construction - keep these in sync
    # if you change one, change the other, or reps/checkpoints won't line up.
    letters = list(string.ascii_lowercase)[:number_pl]
    symbols = ["∧", "¬", "(", ")", " "]
    if tokenization == "bigram":
        alphabet = symbols + letters
        vocab = (["[CLS]"] if use_cls else []) + ["[PAD]", "[MASK]"] + \
                [c1 + c2 for c1 in alphabet for c2 in alphabet]
    else:
        vocab = (["[CLS]"] if use_cls else []) + ["[PAD]", "[MASK]"] + symbols + letters
    tok_to_id = {tok: i for i, tok in enumerate(vocab)}
    return vocab, tok_to_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                         help="name of the folder of the model that'll be visualized")
    parser.add_argument("--epoch", type=str, required=True,
                         help="epoch checkpoint to load")
    parser.add_argument("--dataset_folder", type=str, default=None,
                         help="dataset folder to pull number_pl from; defaults to the "
                              "dataset the model was trained on")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n-sample", type=int, default=None,
                         help="subsample this many tokens before running t-SNE. "
                              "Default: use all tokens that contain exactly one letter.")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    model_folder = project_root / "model" / args.model
    checkpoint = model_folder / f"epoch_{args.epoch}.pt"

    params = pickle.load(open(model_folder / "params.pkl", "rb"))
    dataset_folder_trained_on, cls_model, *_rest = params
    tokenization = _rest[-1] if len(_rest) >= 5 else "char"

    dataset_folder = args.dataset_folder or dataset_folder_trained_on
    ds_params_path = project_root / "dataset" / dataset_folder / "params.pkl"
    number_pl, *_ = pickle.load(open(ds_params_path, "rb"))

    vocab, tok_to_id = build_vocab(number_pl, cls_model, tokenization)

    state_dict = torch.load(checkpoint, map_location="cpu")
    emb = state_dict["tok_emb.weight"].numpy()  # (vocab_size, hidden)

    if emb.shape[0] != len(vocab):
        raise ValueError(
            f"Rebuilt vocab size ({len(vocab)}) doesn't match checkpoint's embedding "
            f"table ({emb.shape[0]}). tokenization={tokenization!r} cls={cls_model} "
            f"number_pl={number_pl} - check these match how the model was trained."
        )

    letters = set(string.ascii_lowercase[:number_pl])
    tokens, tok_labels, rows = [], [], []
    for tok, idx in tok_to_id.items():
        contained = [c for c in tok if c in letters]
        if len(contained) == 1:
            tokens.append(tok)
            tok_labels.append(contained[0])
            rows.append(emb[idx])

    if not rows:
        raise ValueError("No single-letter tokens found - check tokenization/vocab reconstruction.")

    embeddings = np.stack(rows)
    labels = np.array(tok_labels)
    print(f"{len(embeddings)} single-letter tokens found across {len(set(tok_labels))} letters "
          f"(tokenization={tokenization})")

    if args.n_sample is not None and args.n_sample < len(embeddings):
        idx = np.random.RandomState(args.random_state).choice(
            len(embeddings), size=args.n_sample, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        tokens = [tokens[i] for i in idx]

    eff_perplexity = min(args.perplexity, len(embeddings) - 1)
    if eff_perplexity != args.perplexity:
        print(f"reducing perplexity {args.perplexity} -> {eff_perplexity} "
              f"(only {len(embeddings)} points)")

    tsne = TSNE(n_components=2, perplexity=eff_perplexity, random_state=args.random_state,
                init="pca", learning_rate="auto")
    coords = tsne.fit_transform(embeddings)

    unique_letters = sorted(set(labels))
    colors = cm.tab20(np.linspace(0, 1, len(unique_letters)))
    letter_to_color = dict(zip(unique_letters, colors))

    plt.figure(figsize=(9, 8))
    for letter in unique_letters:
        mask = labels == letter
        plt.scatter(coords[mask, 0], coords[mask, 1], color=letter_to_color[letter],
                    label=letter, alpha=0.75, s=25)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.title(f"t-SNE of non-contextual token embeddings, colored by letter\n"
              f"{args.model} (epoch {args.epoch}, {tokenization} tokenization)")
    plt.legend(bbox_to_anchor=(1.02, 1), fontsize=7, ncol=1)

    out_dir = project_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_epoch{args.epoch}_tsne_letters.png"

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved plot to {out_path}  ({len(embeddings)} points, perplexity={eff_perplexity})")


if __name__ == "__main__":
    main()