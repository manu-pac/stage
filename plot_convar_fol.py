import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import training as tr
import tf_generation_fol as tfg
from classes_fol import P
from reps import infer_arch_from_state_dict


def load_model_and_vocab(project_root, model_name, epoch, device):
    model_folder = project_root / "model" / model_name
    checkpoint = model_folder / f"epoch_{epoch}.pt"

    (t_folder_name, cls_model, bs, e, hidden, heads, layers, rb,
     tokenization, logic) = pickle.load(open(model_folder / "params.pkl", "rb"))

    if logic == "pl":
        raise ValueError("unique-witness-variable reps only make sense for FOL models, "
                          f"but this checkpoint was trained on logic={logic!r}")
    if tokenization != "char":
        raise ValueError(f"this script assumes tokenization='char', got {tokenization!r}")

    t_folder = project_root / "dataset" / t_folder_name
    t_params = pickle.load(open(t_folder / "params.pkl", "rb"))

    state_dict = torch.load(checkpoint, map_location=device)
    vocab_size_ckpt, max_len = infer_arch_from_state_dict(state_dict)

    # load the exact vocab saved at training time (more robust than rebuilding it,
    # since it can't drift from whatever training.py actually used)
    vocab = pickle.load(open(model_folder / "vocab.pkl", "rb"))
    tok_to_id = {tok: i for i, tok in enumerate(vocab)}
    letters = [t for t in vocab if t not in
               ("[CLS]", "[PAD]", "[MASK]", "∃", "∧", "¬", "(", ")", ",")]
    if len(vocab) != vocab_size_ckpt:
        raise ValueError(
            f"Saved vocab size ({len(vocab)}) doesn't match checkpoint vocab size ({vocab_size_ckpt})."
        )

    tr.hidden = hidden
    tr.heads = heads
    tr.layers = layers
    tr.vocab_size = vocab_size_ckpt
    tr.max_len = max_len
    tr.pad_id = tok_to_id["[PAD]"]
    tr.mask_id = tok_to_id["[MASK]"]
    tr.cls = cls_model
    tr.tok_to_id = tok_to_id
    tr.vocab = vocab
    tr.letters = letters
    tr.tokenization = tokenization
    tr.tfg = tfg

    model = tr.EncoderTransformer()
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, vocab, tok_to_id, cls_model, max_len, t_folder, t_params


def setup_tfg(r_folder, r_params, active_world, alt_worlds):
    domain, variables = r_params['domain'], r_params['variables']
    predicates_info = r_params['predicates']
    min_arity, max_arity = r_params['min_arity'], r_params['max_arity']
    min_depth, max_depth = r_params['min_depth'], r_params['max_depth']
    predicates = [P(name, arity) for name, arity in predicates_info]

    tfg.setup(domain_=domain, variables_=variables, predicates_=predicates,
              min_arity_=min_arity, max_arity_=max_arity,
              min_depth_=min_depth, max_depth_=max_depth,
              act_world_=active_world, alt_worlds_=alt_worlds)


def extract_records(model, dataloader, positions_by_idx, cls_model, vocab, device, form_le=None):
    """
    Returns a list of dicts: {"idx": int, "var": str, "value": int, "vector": np.ndarray}
    one entry per unique-witness-variable occurrence across the dataset.
    """
    records = []
    cls_offset = 1 if cls_model else 0

    with torch.no_grad():
        for batch_input_ids, batch_attention_mask, batch_idx in dataloader:
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)

            hidden_states, _ = model(batch_input_ids, batch_attention_mask)  # (B, T, H)

            for i, idx in enumerate(batch_idx.tolist()):
                unique_vars = positions_by_idx[idx]  # {var_str: {"value":, "position":}}
                for var_name, info in unique_vars.items():
                    token_idx = info["position"] + cls_offset
                    val = info["value"]

                    tok_id = batch_input_ids[i, token_idx].item()
                    actual_char = vocab[tok_id]
                    if actual_char != var_name:
                        debug_str = f" formula={form_le(idx, 0, [])}" if form_le else ""
                        raise AssertionError(
                            f"Token alignment mismatch for idx={idx}, var={var_name!r}: "
                            f"expected char {var_name!r} at token_idx={token_idx} "
                            f"(position={info['position']}, cls_offset={cls_offset}), "
                            f"but found token decodes to {actual_char!r}.{debug_str}\n"
                            f"This means the char-position -> token-index assumption doesn't "
                            f"hold for this model's encode(); inspect training.py's encode() "
                            f"for the 'char' branch and fix the offset logic above."
                        )

                    vec = hidden_states[i, token_idx, :].detach().cpu().numpy()
                    records.append({"idx": idx, "var": var_name, "value": val, "vector": vec})

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="name of the folder of the model that'll be used")
    parser.add_argument("--epoch", required=True, help="epoch that will be used")
    parser.add_argument("--dataset_folder", required=True,
                         help="dataset folder containing dev_data.pkl (output of plotting_fol.py)")
    parser.add_argument("--alt_world", type=int, default=None)
    parser.add_argument("--color_by", choices=["value", "var"], default="value",
                         help="color t-SNE points by the domain element the variable maps to, "
                              "or by the variable name/letter")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--output", default=None, help="path to save the plot (png). "
                                                         "Defaults to <dataset_folder>_unique_var_tsne.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    project_root = Path(__file__).resolve().parent
    r_folder = project_root / "dataset" / args.dataset_folder

    model, vocab, tok_to_id, cls_model, max_len, t_folder, t_params = load_model_and_vocab(
        project_root, args.model, args.epoch, device
    )

    r_params = pickle.load(open(r_folder / "params.pkl", "rb"))
    assert r_params['domain'] == t_params['domain'], "domain mismatch between dataset and training"
    assert r_params['variables'] == t_params['variables'], "variables mismatch between dataset and training"
    assert r_params['predicates'] == t_params['predicates'], "predicates mismatch between dataset and training"

    act_world = pickle.load(open(r_folder / "act_world.pkl", "rb"))
    alt_worlds = pickle.load(open(r_folder / "alt_worlds.pkl", "rb"))
    active_world = act_world if args.alt_world is None else alt_worlds[args.alt_world]

    setup_tfg(r_folder, r_params, active_world, alt_worlds)

    dev_data_path = r_folder / "dev_data.pkl"
    if not dev_data_path.exists():
        raise FileNotFoundError(
            f"{dev_data_path} not found -- run plotting_fol.py on this dataset folder first "
            f"to produce the filtered dev_data.pkl."
        )
    with open(dev_data_path, "rb") as f:
        dev_data = pickle.load(f)  # list of (idx, {"unique_vars": {...}})

    idx_list = [idx for idx, _ in dev_data]
    positions_by_idx = {idx: result["unique_vars"] for idx, result in dev_data}

    # dev_data.pkl comes from filtering dev_true_indices, so these are all "true" formulas
    dataset = tr.FormulaDataset(idx_list, max_len, t=True)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    from tf_generation_fol import form_le
    records = extract_records(model, dataloader, positions_by_idx, cls_model, vocab, device, form_le=form_le)
    print(f"Extracted {len(records)} unique-variable contextual representations "
          f"from {len(idx_list)} formulas.")

    vectors = np.stack([r["vector"] for r in records], axis=0)

    tsne = TSNE(n_components=2, perplexity=min(args.perplexity, max(5, len(records) // 4)),
                init="pca", random_state=0)
    embedding = tsne.fit_transform(vectors)

    if args.color_by == "value":
        labels = [r["value"] for r in records]
        cmap = "viridis"
    else:
        labels = [r["var"] for r in records]
        unique_labels = sorted(set(labels))
        label_to_int = {lab: i for i, lab in enumerate(unique_labels)}
        labels = [label_to_int[l] for l in labels]
        cmap = "tab10"

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=cmap, s=12, alpha=0.8)
    legend = ax.legend(*scatter.legend_elements(), title=args.color_by, loc="best", fontsize=8)
    ax.add_artist(legend)
    ax.set_title(f"t-SNE of unique-witness-variable contextual reps ({args.model}, epoch {args.epoch})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    out_path = Path(args.output) if args.output else project_root / f"{args.dataset_folder}_unique_var_tsne.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    # also dump raw records + embedding for later reuse (e.g. different coloring/plots)
    records_out_path = out_path.with_suffix(".pkl")
    with open(records_out_path, "wb") as f:
        pickle.dump({"records": records, "embedding": embedding}, f)
    print(f"Saved raw records + embedding to {records_out_path}")


if __name__ == "__main__":
    main()