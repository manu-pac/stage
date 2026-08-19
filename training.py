import pickle
from pathlib import Path
import argparse
import string
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
import random

number_pl = None
cls = None
letters = []
vocab = []
idx_t = []
tok_to_id = {}
max_len = None
vocab_size = None
pad_id = None
mask_id = None
optimizer = None
loss_fn = None
hidden = None
device = None
model = None
dataset = None
dataloader = None
dev_dataset = None
dev_dataloader = None
history = []
best_dev_loss = None
best_epoch = None

# tokenization
BIGRAM2_FILLER = "_"  # placeholder second char for the odd trailing character, bigram2 only

def encode(i, max_len, t=True):
    formula = str(tfg.true_le(i)) if t else str(tfg.false_le(i))
    if tokenization == "bigram":
        toks = [formula[j:j+2] for j in range(len(formula) - 1)]
    elif tokenization == "bigram2":
        toks = [formula[j:j+2] for j in range(0, len(formula), 2)]  # non-overlapping pairs
        if len(toks[-1]) == 1:  # odd length: pad the leftover char with the filler
            toks[-1] = toks[-1] + BIGRAM2_FILLER
    else:
        toks = list(formula)
    ids = [tok_to_id[tok] for tok in toks]
    if cls:
        ids = [tok_to_id["[CLS]"]] + ids
    real_len = len(ids)
    ids += [tok_to_id["[PAD]"]] * (max_len - real_len)
    mask = [1]*real_len + [0]*(max_len - real_len)
    return ids, mask

def seq_len(s):
    if tokenization == "bigram":
        return len(s) - 1
    elif tokenization == "bigram2":
        return -(-len(s) // 2)  # ceil(len(s) / 2)
    else:
        return len(s)

# model
class EncoderTransformer(nn.Module):
  def __init__(self):
    super().__init__()
    self.tok_emb = nn.Embedding(vocab_size, hidden, padding_idx=pad_id) 
    self.pos_emb = nn.Embedding(max_len, hidden) 
    encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=hidden*4,batch_first=True) 
    self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=layers)
    self.mlm_head = nn.Linear(hidden, vocab_size) 

  def forward(self, input_ids, attention_mask):
    B,T= input_ids.shape
    positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
    x = self.tok_emb(input_ids) + self.pos_emb(positions)
    pad_mask = (attention_mask == 0)  # True = ignore this position
    hidden = self.encoder(x, src_key_padding_mask=pad_mask)
    logits = self.mlm_head(hidden)
    return hidden, logits    

def mask_tokens(input_ids, attention_mask, mlm_prob=0.15):
    labels = input_ids.clone()
    prob = torch.full(input_ids.shape, mlm_prob, device=input_ids.device)
    prob[attention_mask == 0] = 0.0
    if cls:
        prob[:, 0] = 0.0   # never mask [CLS]
    mask = torch.bernoulli(prob).bool()
    labels[~mask] = -100
    masked_input = input_ids.clone()
    masked_input[mask] = mask_id
    return masked_input, labels

def train_step(input_ids, attention_mask):
    masked_input, labels = mask_tokens(input_ids, attention_mask)
    _, logits = model(masked_input, attention_mask)
    loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

class FormulaDataset(torch.utils.data.Dataset):
    def __init__(self, idx_list, max_len, t=True):
        self.idx_list = idx_list
        self.max_len = max_len
        self.t = t

    def __len__(self):
        return len(self.idx_list)

    # encoding during training
    def __getitem__(self,idx):
        ids, mask = encode(self.idx_list[idx], self.max_len, self.t)
        return torch.tensor(ids), torch.tensor(mask), idx

# training
def eval_loss(dataloader):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch_masked_input, batch_labels, batch_attention_mask in dataloader:
            batch_masked_input = batch_masked_input.to(device)
            batch_labels = batch_labels.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            _, logits = model(batch_masked_input, batch_attention_mask)
            loss = loss_fn(logits.view(-1, vocab_size), batch_labels.view(-1))
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / n_batches


def main():
    print("starting")
    parser = argparse.ArgumentParser()
    parser.add_argument("--logic", choices=["pl","fol"], default="pl",
                    help="choose logic: pl or fol")
    parser.add_argument("--dataset_folder", type=str, required=True, help="name of the folder where the dataset that'll be used for training is stored")
    parser.add_argument("--cls", action="store_true", help="add [CLS] on tokenization")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--r_bias", action="store_true", help="add reporting bias to the creation of the batches")
    parser.add_argument("--bias_power", type=float, default=1.0, help="exponent applied to prob() before it's used as a sampling weight (only when --r_bias is set); >1 sharpens the contrast between high- and low-informativity formulas, 1.0 = original behavior")
    parser.add_argument("--seed", type=int, default=0, help="seed for model init and sampling")
    parser.add_argument("--patience", type=int, default=5, help="stop after this many consecutive epochs with no dev loss improvement; halves lr on each bad epoch")
    parser.add_argument("--tokenization", choices=["char","bigram","bigram2"], default="char") 
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    global hidden, heads, layers, idx_t, dev_t, number_pl, cls, max_len, vocab, letters, tok_to_id, vocab_size, pad_id, mask_id, tokenization, tfg

    folder = Path(__file__).resolve().parent / "dataset" / args.dataset_folder

    cls = args.cls
    tokenization = args.tokenization
    batch_size = args.batch_size
    epochs = args.epochs
    hidden = args.hidden
    heads = args.heads
    layers = args.layers

    # Import the right generator module based on logic
    logic = args.logic
    if logic == 'pl':
        import tf_generation as tfg
    else:
        import tf_generation_fol as tfg
        from classes_fol import P   # needed to rebuild predicate objects

    # load the parameters the corpus was generated with
    act_world = pickle.load(open(folder / "act_world.pkl", "rb"))
    alt_worlds = pickle.load(open(folder / "alt_worlds.pkl", "rb"))
    params = pickle.load(open(folder / "params.pkl", "rb"))

    # Handle parameter loading for PL vs FOL
    if logic == 'pl':
        number_pl, min_depth, max_depth, corpus_size, prop_td, n_worlds = params
        # PL setup
        tfg.setup(number_pl_=number_pl, max_depth_=max_depth, 
                  act_world_=act_world, alt_worlds_=alt_worlds)
        # Build alphabet for PL
        letters = list(string.ascii_lowercase)[:number_pl]
        symbols = ["∧", "¬", "(", ")"]
    else:
        # FOL: params is a dictionary
        domain = params['domain']
        variables = params['variables']
        predicates_info = params['predicates']   # list of (name, arity)
        min_arity = params['min_arity']
        max_arity = params['max_arity']
        min_depth = params['min_depth']
        max_depth = params['max_depth']
        corpus_size = params['corpus_size']
        n_worlds = params['n_worlds']
        number_pr = params['number_pr']
        number_vr = params['number_vr']
        
        # Rebuild P objects (required for setup)
        predicates = [P(name, arity) for name, arity in predicates_info]
        
        # FOL setup
        tfg.setup(domain_=domain, variables_=variables, predicates_=predicates,
                  min_arity_=min_arity, max_arity_=max_arity,
                  min_depth_=min_depth, max_depth_=max_depth,
                  act_world_=act_world, alt_worlds_=alt_worlds)
        
        # Build alphabet for FOL
        pred_letters = list(string.ascii_uppercase[:number_pr])
        var_letters = list(string.ascii_lowercase[:number_vr])
        symbols = ["∃", "∧", "¬", "(", ")", ","]
        letters = pred_letters + var_letters
        
        # We don't need number_pl for FOL, but keep it as None for compatibility
        number_pl = None

    # load the actual corpus
    idx_t = pickle.load(open(folder / "train.pkl", "rb"))
    dev_t = pickle.load(open(folder / "dev_t.pkl", "rb"))
    dev_f = pickle.load(open(folder / "dev_f.pkl", "rb"))

    # build pieces for tokenization (using the appropriate alphabet)
    alphabet = symbols + letters
    if tokenization == "bigram":
        vocab = (["[CLS]"] if cls else []) + ["[PAD]","[MASK]"] + [c1+c2 for c1 in alphabet for c2 in alphabet]
    elif tokenization == "bigram2":
        vocab = (["[CLS]"] if cls else []) + ["[PAD]","[MASK]"] + \
                [c1+c2 for c1 in alphabet for c2 in alphabet] + \
                [c + BIGRAM2_FILLER for c in alphabet]  # padded tokens for odd trailing chars
    else:
        # For char-level tokenization, use the appropriate symbols
        if logic == 'pl':
            vocab = (["[CLS]"] if cls else []) + ["[PAD]","[MASK]","∧","¬","(",")"] + letters
        else:
            vocab = (["[CLS]"] if cls else []) + ["[PAD]","[MASK]","∃","∧","¬","(",")",","] + letters
    
    tok_to_id = {tok: i for i, tok in enumerate(vocab)}
    
    # Compute max_len (for FOL we need to handle the string conversion appropriately)
    if logic == 'pl':
        max_len = max(max(seq_len(str(tfg.true_le(i))) for i in idx_t),
                  max(seq_len(str(tfg.true_le(i))) for i in dev_t),
                  max(seq_len(str(tfg.false_le(i))) for i in dev_f))
    else:
        # For FOL, true_le and false_le are aliases to form_le
        max_len = max(max(seq_len(str(tfg.true_le(i))) for i in idx_t),
                  max(seq_len(str(tfg.true_le(i))) for i in dev_t),
                  max(seq_len(str(tfg.false_le(i))) for i in dev_f))
    
    if cls:
        max_len += 1
    vocab_size = len(vocab)
    pad_id = tok_to_id["[PAD]"]
    mask_id = tok_to_id["[MASK]"]

    # create model, optimizer and loss function
    global model, optimizer, loss_fn, dataloader, device, dataset, dev_dataset, dev_dataloader
    model = EncoderTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    dataset = FormulaDataset(idx_t, max_len)

    # training
    if args.r_bias:
        probs = pickle.load(open(folder / "probs.pkl", "rb"))
        probs_t = torch.tensor(probs, dtype=torch.float) ** args.bias_power
    else:
        probs_t = torch.ones(len(dataset)) #if flag --bias is not activated, the random sampling happens uniformly 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.5, device=0)

    model = model.to(device)

    sampler = WeightedRandomSampler(weights=probs_t, num_samples=len(dataset), replacement=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    dev_dataset = FormulaDataset(dev_t, max_len)
    dev_dataloader_raw = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    # freeze dev masking pattern once, same for every run/epoch
    torch.manual_seed(args.seed)  # reuse the same fixed seed so both conditions see identical dev masking
    fixed_dev_batches = []
    with torch.no_grad():
        for batch_input_ids, batch_attention_mask, _ in dev_dataloader_raw:
            masked_input, labels = mask_tokens(batch_input_ids, batch_attention_mask)
            fixed_dev_batches.append((masked_input, labels, batch_attention_mask))

    # wrap it as something eval_loss can iterate the same way every time
    dev_dataloader = fixed_dev_batches
    
    # folders to save the checkpoints
    project_root = Path(__file__).resolve().parent
    suffix = ""
    if args.cls:
        suffix += "_cls"
    if args.r_bias:
        suffix += "_rb"
        if args.bias_power != 1.0:
            suffix += f"_bp{args.bias_power:g}"
    if tokenization == "bigram":
        suffix += "_bigram"
    elif tokenization == "bigram2":
        suffix += "_bigram2"
    # Add logic to suffix
    suffix += f"_{logic}"

    out_dir = (project_root/ "model"/ f"{args.batch_size}bs_{args.epochs}e_{args.hidden}hl_{args.heads}h_{args.layers}l{suffix}_seed{args.seed}_{args.dataset_folder}")
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_dev_loss = float("inf")
    best_epoch = None
    bad_epochs = 0

    seen = set()
    total_seen = 0

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch_input_ids, batch_attention_mask, batch_idx in dataloader:
            batch_size = len(batch_idx)
            total_seen += batch_size
            seen.update(batch_idx.tolist())

            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            loss = train_step(batch_input_ids, batch_attention_mask)
            total_loss += loss
            n_batches += 1
        train_loss = total_loss / n_batches
        dev_loss = eval_loss(dev_dataloader)
        history.append((epoch, train_loss, dev_loss))
        print(f"epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  dev_loss={dev_loss:.4f}")

        torch.save(model.state_dict(), out_dir / f"epoch_{epoch + 1}.pt")

        if dev_loss < best_dev_loss - 1e-3:
            best_dev_loss = dev_loss
            best_epoch = epoch + 1
            bad_epochs = 0
        else:
            bad_epochs += 1
            for g in optimizer.param_groups:  # anneal lr, same pattern as probe.py
                g["lr"] *= 0.5
            if bad_epochs >= args.patience:
                print(f"no dev improvement for {args.patience} epochs, stopping at epoch {epoch + 1}")
                break

    print(f"best epoch: {best_epoch}  best_dev_loss: {best_dev_loss:.4f}")
    print(f"stopped at epoch: {epoch + 1}")
    print(f"Total examples seen: {total_seen}")
    print(f"Unique examples seen: {len(seen)}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Coverage: {100 * len(seen) / len(dataset):.2f}%")

    p_out = out_dir / "params.pkl"
    with open(p_out, "wb") as f:
        pickle.dump((args.dataset_folder, cls, batch_size, epochs, hidden, heads, layers, args.r_bias, tokenization, logic), f)

    with open(out_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
        
    # plot losses
    epochs_plot = [h[0]+1 for h in history]
    train_losses = [h[1] for h in history]
    dev_losses = [h[2] for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_plot, train_losses, label='Train Loss')
    plt.plot(epochs_plot, dev_losses, label='Dev Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Dev Losses')
    plt.grid(True)
    plt.legend()

    plt.savefig(out_dir / "loss_plot.png")
    plt.show()

if __name__ == "__main__":
    main()