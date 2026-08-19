import argparse
import pickle
import string
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import training as tr


def infer_arch_from_state_dict(state_dict):
    vocab_size = state_dict["tok_emb.weight"].shape[0]
    max_len = state_dict["pos_emb.weight"].shape[0]
    return vocab_size, max_len


def build_vocab(logic, use_cls, tokenization="char",
                 number_pl=None, number_pr=None, number_vr=None):
    # builds the same vocab training.py would have built for this logic/tokenization
    if logic == "pl":
        letters = list(string.ascii_lowercase)[:number_pl]
        symbols = ["∧", "¬", "(", ")"]
    else:
        pred_letters = list(string.ascii_uppercase)[:number_pr]
        var_letters = list(string.ascii_lowercase)[:number_vr]
        letters = pred_letters + var_letters
        symbols = ["∃", "∧", "¬", "(", ")", ","]

    alphabet = symbols + letters
    if tokenization == "bigram":
        vocab = (["[CLS]"] if use_cls else []) + ["[PAD]", "[MASK]"] + \
                [c1 + c2 for c1 in alphabet for c2 in alphabet]
    elif tokenization == "bigram2":
        vocab = (["[CLS]"] if use_cls else []) + ["[PAD]", "[MASK]"] + \
                [c1 + c2 for c1 in alphabet for c2 in alphabet] + \
                [c + tr.BIGRAM2_FILLER for c in alphabet]
    else:
        vocab = (["[CLS]"] if use_cls else []) + ["[PAD]", "[MASK]"] + symbols + letters

    tok_to_id = {tok: i for i, tok in enumerate(vocab)}
    return vocab, tok_to_id, letters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="name of the folder of the model that'll be used")
    parser.add_argument("--epoch", help="epoch that will be used")
    parser.add_argument("--dataset_name", help="dataset to get the reps from (e.g. dev_t, dev_f, train)")
    parser.add_argument("--dataset_folder", default=None, help="folder the dataset that will be represented is located on")
    parser.add_argument("--cls", action="store_true", help="get cls reps")
    parser.add_argument("--alt_world", type=int, default=None)
    args = parser.parse_args()
    # TODO: TROCAR --CLS POR TYPE (E MUDAR DAQ PRA FRENTE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    project_root = Path(__file__).resolve().parent
    model_folder = project_root / "model" / args.model
    checkpoint = model_folder / f"epoch_{args.epoch}.pt"

    (t_folder_name, cls_model, bs, e, hidden, heads, layers, rb,
     tokenization, logic) = pickle.load(open(model_folder / "params.pkl", "rb"))

    # pick the generator module that matches how this model was trained
    if logic == "pl":
        import tf_generation as tfg
    else:
        import tf_generation_fol as tfg
        from classes_fol import P

    t_folder = project_root / "dataset" / t_folder_name  # folder of the dataset the model was trained in

    dataset_folder = args.dataset_folder if args.dataset_folder is not None else t_folder_name
    r_folder = project_root / "dataset" / dataset_folder  # folder of dataset that will be represented

    # reload the world/params the corpus (and hence vocab) was built with
    act_world = pickle.load(open(r_folder / "act_world.pkl", "rb"))
    alt_worlds = pickle.load(open(r_folder / "alt_worlds.pkl", "rb"))
    r_params = pickle.load(open(r_folder / "params.pkl", "rb"))
    t_params = pickle.load(open(t_folder / "params.pkl", "rb"))

    if logic == "pl":
        number_pl, min_depth, max_depth, corpus_size, prop_td, n_worlds = r_params
        t_number_pl, _, _, _, _, _ = t_params
        assert number_pl == t_number_pl, (
            f"target dataset's number_pl ({number_pl}) doesn't match training's ({t_number_pl})"
        )
        number_pr = number_vr = None
        predicates = None
    else:
        domain, variables = r_params['domain'], r_params['variables']
        predicates_info = r_params['predicates']
        min_arity, max_arity = r_params['min_arity'], r_params['max_arity']
        min_depth, max_depth = r_params['min_depth'], r_params['max_depth']
        number_pr, number_vr = r_params['number_pr'], r_params['number_vr']

        t_domain, t_variables = t_params['domain'], t_params['variables']
        t_predicates_info = t_params['predicates']

        assert domain == t_domain, (
            f"target dataset's domain ({domain}) doesn't match training's ({t_domain})"
        )
        assert variables == t_variables, (
            f"target dataset's variables ({variables}) don't match training's ({t_variables})"
        )
        assert predicates_info == t_predicates_info, (
            f"target dataset's predicates ({predicates_info}) don't match training's ({t_predicates_info})"
        )

        number_pl = None
        predicates = [P(name, arity) for name, arity in predicates_info]

    if args.alt_world is None:
        active_world = act_world
    else:
        active_world = alt_worlds[args.alt_world]

    if logic == "pl":
        tfg.setup(number_pl_=number_pl, max_depth_=max_depth,
                  act_world_=active_world, alt_worlds_=alt_worlds)
    else:
        tfg.setup(domain_=domain, variables_=variables, predicates_=predicates,
                  min_arity_=min_arity, max_arity_=max_arity,
                  min_depth_=min_depth, max_depth_=max_depth,
                  act_world_=active_world, alt_worlds_=alt_worlds)

    r_set_path = r_folder / f"{args.dataset_name}.pkl"  # path of dataset that will be represented

    set_t = False if (args.dataset_name == "dev_f") else True

    # load checkpoint and figure out what we can straight from its weights
    state_dict = torch.load(checkpoint, map_location=device)
    vocab_size_ckpt, max_len = infer_arch_from_state_dict(state_dict)

    use_cls = args.cls

    if use_cls and not cls_model:
        raise ValueError("Requested --cls output, but this model wasn't trained with a CLS token.")

    vocab, tok_to_id, letters = build_vocab(
        logic, cls_model, tokenization,
        number_pl=number_pl, number_pr=number_pr, number_vr=number_vr
    )

    if len(vocab) != vocab_size_ckpt:
        raise ValueError(
            f"Built vocab size ({len(vocab)}, use_cls={cls_model}) doesn't match the checkpoint's "
            f"vocab size ({vocab_size_ckpt})."
        )

    print(f"logic={logic} model: hidden={hidden} layers={layers} max_len={max_len} "
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
    tr.letters = letters
    tr.tokenization = tokenization
    tr.tfg = tfg  # training.py's encode()/true_le()/false_le() read this global directly

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

    suffix = f"_{args.alt_world}" if args.alt_world is not None else ""

    out_path = project_root / "reps" / args.model / args.epoch / "mean" / dataset_folder / f"{args.dataset_name}{suffix}"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump({"indexes": idx_list, "type": "mean", "reps": mean_reps}, f)
    print(f"saved mean representations for {len(idx_list)} items to {out_path}")
    print(f"mean shape: {mean_reps.shape}")

    if use_cls:
        cls_reps = np.concatenate(cls_reps, axis=0)
        cls_path = project_root / "reps" / args.model / args.epoch / "cls" / dataset_folder / f"{args.dataset_name}{suffix}"
        cls_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cls_path, "wb") as f:
            pickle.dump({"indexes": idx_list, "type": "cls", "reps": cls_reps}, f)
        print(f"saved cls representations for {len(idx_list)} items to {cls_path}")
        print(f"cls shape: {cls_reps.shape}")


if __name__ == "__main__":
    main()