"""
    python get_representations.py \
        --checkpoint #model ill use \
        --set-path #path to the dataset i'll get the reps to \
        --folder-name #dataset folder name used at training time, needed to reload params.pkl / act_world.pkl / alt_worlds.pkl \
        --output #pth where reps should be sved \
        --cls-repr # do cls representation
        --cls #if the model was trined using cls
        --set_f # if the set is made out of ormulas tht are false on the actual world
"""

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import tf_generation as tfg
import training as tr

def infer_arch_from_state_dict(state_dict):
    hidden = state_dict["tok_emb.weight"].shape[1]
    vocab_size = state_dict["tok_emb.weight"].shape[0]
    max_len = state_dict["pos_emb.weight"].shape[0]

    layer_ids = set()
    pattern = re.compile(r"encoder\.layers\.(\d+)\.")
    for key in state_dict.keys():
        m = pattern.match(key)
        if m:
            layer_ids.add(int(m.group(1)))
    layers = max(layer_ids) + 1 if layer_ids else None

    return hidden, vocab_size, max_len, layers


def build_vocab(number_pl, use_cls):
    letters = list(__import__("string").ascii_lowercase)[:number_pl]
    vocab = (["[CLS]"] if use_cls else []) + ["[PAD]", "[MASK]", "∧", "¬", "(", ")", " "] + letters
    tok_to_id = {tok: i for i, tok in enumerate(vocab)}
    return vocab, tok_to_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,)
    parser.add_argument("--set-path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--folder-name", type=str, required=True)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cls-repr", action="store_true")
    parser.add_argument("--cls", action="store_true")
    parser.add_argument("--set_f", default=True, action="store_false")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    project_root = Path(__file__).resolve().parent
    folder = project_root / "dataset" / args.folder_name

    # reload the world/params the corpus (and hence vocab) was built with
    act_world = pickle.load(open(folder / "act_world.pkl", "rb"))
    alt_worlds = pickle.load(open(folder / "alt_worlds.pkl", "rb"))
    number_pl, min_depth, max_depth, corpus_size, prop_td, n_worlds = pickle.load(
        open(folder / "params.pkl", "rb")
    )
    tfg.setup(number_pl_=number_pl, max_depth_=max_depth, act_world_=act_world, alt_worlds_=alt_worlds)
 
    # load checkpoint and figure out what we can straight from its weights
    state_dict = torch.load(args.checkpoint, map_location=device)
    hidden, vocab_size_ckpt, max_len, layers = infer_arch_from_state_dict(state_dict)

    use_cls = args.cls
 
    vocab, tok_to_id = build_vocab(number_pl, use_cls)
    if len(vocab) != vocab_size_ckpt:
        raise ValueError(
            f"Built vocab size ({len(vocab)}, use_cls={use_cls}) doesn't match the checkpoint's "
            f"vocab size ({vocab_size_ckpt})."
        )
 
    print(f"inferred: hidden={hidden} layers={layers} max_len={max_len} "
          f"vocab_size={vocab_size_ckpt} use_cls={use_cls} heads={args.heads}")
 
    # populate train.py's module-level globals so its EncoderTransformer/encode/
    # FormulaDataset behave exactly as they did at training time
    tr.hidden = hidden
    tr.heads = args.heads
    tr.layers = layers
    tr.vocab_size = vocab_size_ckpt
    tr.max_len = max_len
    tr.pad_id = tok_to_id["[PAD]"]
    tr.mask_id = tok_to_id["[MASK]"]
    tr.cls = use_cls
    tr.tok_to_id = tok_to_id
    tr.vocab = vocab
    tr.letters = list(__import__("string").ascii_lowercase)[:number_pl]
 
    model = tr.EncoderTransformer()
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
 
    # load the target set and build a loader (reusing train.py's FormulaDataset/encode)
    idx_list = pickle.load(open(args.set_path, "rb"))
    dataset = tr.FormulaDataset(idx_list, max_len, t=args.set_f)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
 
    mean_reps = []
    cls_reps = [] if args.cls_repr else None
 
    with torch.no_grad():
        for batch_input_ids, batch_attention_mask in dataloader:
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
 
            hidden_states, _ = model(batch_input_ids, batch_attention_mask)  # (B, T, H)
 
            mask = batch_attention_mask.unsqueeze(-1).float()  # (B, T, 1)
            if use_cls:
                # don't let [CLS] dominate the mean-pool; exclude position 0 from mean
                pooled_mask = mask.clone()
                pooled_mask[:, 0, :] = 0.0
            else:
                pooled_mask = mask
 
            summed = (hidden_states * pooled_mask).sum(dim=1)
            counts = pooled_mask.sum(dim=1).clamp(min=1e-9)
            mean_batch = summed / counts
            mean_reps.append(mean_batch.cpu().numpy())
 
            if args.cls_repr:
                cls_reps.append(hidden_states[:, 0, :].cpu().numpy())
 
    mean_reps = np.concatenate(mean_reps, axis=0)
    
    out_dir = project_root / "reps" / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_stem = Path(args.checkpoint).stem
    set_stem = Path(args.set_path).stem
 
    mean_path = out_dir / f"{ckpt_stem}__{set_stem}__mean.pkl"
    with open(mean_path, "wb") as f:
        pickle.dump({"indexes": idx_list, "type": "mean", "reps": mean_reps}, f)
    print(f"saved mean representations for {len(idx_list)} items to {mean_path}")
    print(f"mean shape: {mean_reps.shape}")
 
    if args.cls_repr:
        cls_reps = np.concatenate(cls_reps, axis=0)
        cls_path = out_dir / f"{ckpt_stem}__{set_stem}__cls.pkl"
        with open(cls_path, "wb") as f:
            pickle.dump({"indexes": idx_list, "type": "cls", "reps": cls_reps}, f)
        print(f"saved cls representations for {len(idx_list)} items to {cls_path}")
        print(f"cls shape: {cls_reps.shape}")
 
if __name__ == "__main__":
    main()