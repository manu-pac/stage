import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import tf_generation as tfg
import training as tr

def infer_arch_from_state_dict(state_dict):
    vocab_size = state_dict["tok_emb.weight"].shape[0]
    max_len = state_dict["pos_emb.weight"].shape[0]

    return vocab_size, max_len


def build_vocab(number_pl, use_cls):
    letters = list(__import__("string").ascii_lowercase)[:number_pl]
    vocab = (["[CLS]"] if use_cls else []) + ["[PAD]", "[MASK]", "∧", "¬", "(", ")", " "] + letters
    tok_to_id = {tok: i for i, tok in enumerate(vocab)}
    return vocab, tok_to_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="name of the model")
    parser.add_argument("--epoch", help="epoch that will be used")
    parser.add_argument("--dataset_name", help="dataset to get the reps from (e.g. dev_t, dev_f, train)")
    parser.add_argument("--dataset_folder", default=None, help="folder the dataset that will be represented is located on")
    parser.add_argument("--cls", action="store_true", help="get cls reps")
    args = parser.parse_args()
    # TODO: TROCAR --CLS POR TYPE (E MUDAR DAQ PRA FRENTE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    project_root = Path(__file__).resolve().parent
    model_folder = project_root / "model" / args.model
    checkpoint = model_folder / f"epoch_{args.epoch}.pt" 
    t_folder_name, cls_model, bs, e, hidden, heads, layers, rb = pickle.load(open(model_folder / "params.pkl", "rb"))

    t_folder = project_root / "dataset" / t_folder_name # folder of the dataset the model was trained in

    if args.dataset_folder is None:
        dataset_folder = t_folder_name
    else:
        dataset_folder = args.dataset_folder

    r_folder = project_root / "dataset" / dataset_folder  # folder of dataset that will be represented

    r_set_path = r_folder / f"{args.dataset_name}.pkl" # path of dataset that will be represented

    set_t = False if (args.dataset_name=="dev_f") else True 

    # reload the world/params the corpus (and hence vocab) was built with
    act_world = pickle.load(open(r_folder / "act_world.pkl", "rb"))
    alt_worlds = pickle.load(open(r_folder / "alt_worlds.pkl", "rb"))
    number_pl, min_depth, max_depth, corpus_size, prop_td, n_worlds = pickle.load(
        open(r_folder / "params.pkl", "rb")
    )
    tfg.setup(number_pl_=number_pl, max_depth_=max_depth, act_world_=act_world, alt_worlds_=alt_worlds)
 
    # load checkpoint and figure out what we can straight from its weights
    state_dict = torch.load(checkpoint, map_location=device)
    vocab_size_ckpt, max_len = infer_arch_from_state_dict(state_dict)

    use_cls = args.cls

    if use_cls and not cls_model:
        raise ValueError("Requested --cls output, but this model wasn't trained with a CLS token.")

    t_number_pl, _, _, _, _, _ = pickle.load(open(t_folder/"params.pkl", "rb"))

    assert number_pl == t_number_pl, (f"target dataset's number_pl doesn't match training's")

    vocab, tok_to_id = build_vocab(t_number_pl, cls_model)

    if len(vocab) != vocab_size_ckpt:
        raise ValueError(
            f"Built vocab size ({len(vocab)}, use_cls={cls_model}) doesn't match the checkpoint's "
            f"vocab size ({vocab_size_ckpt})."
        )
 
    print(f"model: hidden={hidden} layers={layers} max_len={max_len} "
          f"vocab_size={vocab_size_ckpt} cls_model={cls_model} heads={heads}")
 
    # populate train.py's module-level globals so its EncoderTransformer/encode/
    # FormulaDataset behave exactly as they did at training time
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
    tr.letters = list(__import__("string").ascii_lowercase)[:t_number_pl] 

    model = tr.EncoderTransformer()
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
 
    # load the target set and build a loader (reusing train.py's FormulaDataset/encode)
    idx_list = pickle.load(open(r_set_path, "rb"))
    dataset = tr.FormulaDataset(idx_list, max_len, t=set_t)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
 
    mean_reps = []
    cls_reps = [] if use_cls else None
 
    with torch.no_grad():
        for batch_input_ids, batch_attention_mask, batch_idx in dataloader:
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
 
            hidden_states, _ = model(batch_input_ids, batch_attention_mask)  # (B, T, H)
 
            mask = batch_attention_mask.unsqueeze(-1).float()  # (B, T, 1)
            if cls_model:
                # don't let [CLS] dominate the mean-pool; exclude position 0 from mean
                pooled_mask = mask.clone()
                pooled_mask[:, 0, :] = 0.0
            else:
                pooled_mask = mask
 
            summed = (hidden_states * pooled_mask).sum(dim=1)
            counts = pooled_mask.sum(dim=1).clamp(min=1e-9)
            mean_batch = summed / counts
            mean_reps.append(mean_batch.cpu().numpy())
 
            if use_cls:
                cls_reps.append(hidden_states[:, 0, :].cpu().numpy())
 
    mean_reps = np.concatenate(mean_reps, axis=0)

    out_path = project_root / "reps" / args.model / args.epoch / "mean" / dataset_folder / args.dataset_name

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump({"indexes": idx_list, "type": "mean", "reps": mean_reps}, f)
    print(f"saved mean representations for {len(idx_list)} items to {out_path}")
    print(f"mean shape: {mean_reps.shape}")

    if use_cls:
        cls_reps = np.concatenate(cls_reps, axis=0)
        cls_path = project_root / "reps" / args.model / args.epoch / "cls" / dataset_folder / args.dataset_name
        cls_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cls_path, "wb") as f:
            pickle.dump({"indexes": idx_list, "type": "cls", "reps": cls_reps}, f)
        print(f"saved cls representations for {len(idx_list)} items to {cls_path}")
        print(f"cls shape: {cls_reps.shape}")
 
if __name__ == "__main__":
    main()