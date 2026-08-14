import argparse
import pickle
import string
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE

import tf_generation as tfg

BIGRAM2_FILLER = "_"  # must match training.py's BIGRAM2_FILLER


def build_vocab(number_pl, use_cls, tokenization="bigram"):
    # mirrors training.py's bigram/bigram2 vocab construction
    letters = list(string.ascii_lowercase)[:number_pl]
    symbols = ["∧", "¬", "(", ")", " "]
    alphabet = symbols + letters
    vocab = (["[CLS]"] if use_cls else []) + ["[PAD]", "[MASK]"] + \
            [c1 + c2 for c1 in alphabet for c2 in alphabet]
    if tokenization == "bigram2":
        vocab += [c + BIGRAM2_FILLER for c in alphabet]
    tok_to_id = {tok: i for i, tok in enumerate(vocab)}
    return vocab, tok_to_id


def formula_tokens(s, tokenization):
    # mirrors training.py's encode() chunking exactly (minus [CLS]/padding,
    # which don't matter for just checking which tokens get produced)
    if tokenization == "bigram2":
        toks = [s[j:j + 2] for j in range(0, len(s), 2)]
        if len(toks[-1]) == 1:
            toks[-1] = toks[-1] + BIGRAM2_FILLER
        return toks
    else:  # "bigram" (overlapping)
        return [s[j:j + 2] for j in range(len(s) - 1)]


def tokens_used_in_corpus(dataset_folder_path, number_pl, max_depth, act_world, alt_worlds, tokenization):
    # scans train.pkl/dev_t.pkl/dev_f.pkl (the same files training.py loads),
    # rebuilds each formula's string via tf_generation, and tokenizes it the
    # exact same way encode() does - so this tells us which TOKENS were
    # actually produced during training, not just which letter characters
    # happen to appear somewhere in a formula. This distinction matters a lot
    # for bigram2: e.g. a corpus with min_depth>=2 never has a bare letter as
    # the last character of a formula (Neg/Conj always end in ")"), so a
    # letter's "_"-filler token can go completely unused even though the
    # letter itself appears constantly - checking letter presence alone would
    # wrongly include that never-trained filler token in the plot.
    tfg.setup(number_pl_=number_pl, max_depth_=max_depth, act_world_=act_world, alt_worlds_=alt_worlds)
    seen = set()
    for fname, is_true in (("train.pkl", True), ("dev_t.pkl", True), ("dev_f.pkl", False)):
        fpath = dataset_folder_path / fname
        if not fpath.exists():
            continue
        idx_list = pickle.load(open(fpath, "rb"))
        for i in idx_list:
            formula_str = str(tfg.true_le(i)) if is_true else str(tfg.false_le(i))
            seen.update(formula_tokens(formula_str, tokenization))
    return seen


def valid_bigrams(letters, tokenization="bigram"):
    # every letter can only ever be preceded by "(", "¬", or " ", and only ever
    # followed by ")" or " " (see Neg/Conj __str__) - so these 5 bigrams per
    # letter are the complete, exact set; nothing is approximate here.
    # Under bigram2 (non-overlapping), only every other adjacent pair survives
    # as an actual token, so not all 5 are guaranteed to occur for a given
    # letter - but a letter landing on an odd boundary instead gets padded
    # with the filler token, which never occurs under plain overlapping
    # bigram tokenization. So for bigram2 we keep the same 5 candidates
    # (still the only ones that can ever occur) and add the filler token.
    letter_of = {}
    for letter in letters:
        for tok in (f"¬{letter}", f"({letter}", f" {letter}", f"{letter} ", f"{letter})"):
            letter_of[tok] = letter
        if tokenization == "bigram2":
            letter_of[f"{letter}{BIGRAM2_FILLER}"] = letter
    return letter_of


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                         help="name of the folder of the model that'll be visualized")
    parser.add_argument("--epoch", type=str, required=True,
                         help="epoch checkpoint to load")
    parser.add_argument("--dataset_folder", type=str, default=None,
                         help="dataset folder to pull number_pl from; defaults to the "
                              "dataset the model was trained on")
    parser.add_argument("--perplexity", type=float, default=4,
                         help="fixed at 4 by default: each letter has exactly 5 bigram "
                              "tokens under this grammar (6 under bigram2), and "
                              "perplexity should stay below that")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--all_letters", action="store_true",
                         help="plot every candidate token up to number_pl, even ones "
                              "whose exact token never actually occurs in this dataset's "
                              "train/dev_t/dev_f corpus (default: only plot tokens that "
                              "were actually produced during tokenization)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    model_folder = project_root / "model" / args.model
    checkpoint = model_folder / f"epoch_{args.epoch}.pt"

    params = pickle.load(open(model_folder / "params.pkl", "rb"))
    dataset_folder_trained_on, cls_model, *_rest = params
    tokenization = _rest[-1] if len(_rest) >= 7 else "bigram"

    dataset_folder = args.dataset_folder or dataset_folder_trained_on
    ds_folder_path = project_root / "dataset" / dataset_folder
    ds_params_path = ds_folder_path / "params.pkl"
    number_pl, min_depth, max_depth, corpus_size, prop_td, n_worlds = pickle.load(open(ds_params_path, "rb"))

    vocab, tok_to_id = build_vocab(number_pl, cls_model, tokenization)

    state_dict = torch.load(checkpoint, map_location="cpu")
    emb = state_dict["tok_emb.weight"].numpy()  # (vocab_size, hidden)

    if emb.shape[0] != len(vocab):
        raise ValueError(
            f"Rebuilt vocab size ({len(vocab)}) doesn't match checkpoint's embedding "
            f"table ({emb.shape[0]}). cls={cls_model} number_pl={number_pl} "
            f"tokenization={tokenization!r} - check these match how the model was trained."
        )

    all_letters = list(string.ascii_lowercase)[:number_pl]
    letter_of = valid_bigrams(all_letters, tokenization)

    if not args.all_letters:
        act_world = pickle.load(open(ds_folder_path / "act_world.pkl", "rb"))
        alt_worlds = pickle.load(open(ds_folder_path / "alt_worlds.pkl", "rb"))
        seen_tokens = tokens_used_in_corpus(ds_folder_path, number_pl, max_depth,
                                             act_world, alt_worlds, tokenization)
        before = len(letter_of)
        letter_of = {tok: letter for tok, letter in letter_of.items() if tok in seen_tokens}
        n_letters = len({l for l in letter_of.values()})
        print(f"{len(letter_of)}/{before} candidate tokens actually occur in "
              f"{dataset_folder}'s train/dev_t/dev_f corpus, covering "
              f"{n_letters}/{len(all_letters)} letters - only plotting those "
              f"(pass --all_letters to plot every candidate token regardless)")

    tokens, tok_labels, rows = [], [], []
    for tok, letter in letter_of.items():
        if tok not in tok_to_id:
            continue  # shouldn't happen if build_vocab matches training, but don't crash if it does
        tokens.append(tok)
        tok_labels.append(letter)
        rows.append(emb[tok_to_id[tok]])

    embeddings = np.stack(rows)
    labels = np.array(tok_labels)
    print(f"{len(embeddings)} single-letter tokens found across {len(set(tok_labels))} letters")

    tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=args.random_state,
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
    for (x, y), tok in zip(coords, tokens):
        plt.annotate(tok, (x, y), fontsize=6, alpha=0.8,
                     xytext=(3, 3), textcoords="offset points")
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
    print(f"saved plot to {out_path}  ({len(embeddings)} points, perplexity={args.perplexity})")


if __name__ == "__main__":
    main()