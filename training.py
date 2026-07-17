import pickle
from pathlib import Path
import tf_generation as tfg
import argparse
import string
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
from torch.utils.data import WeightedRandomSampler


## tokenization

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

#parameters
hidden = None

# tokenization

def encode(i,max_len):
    formula = str(tfg.true_le(i))
    ids = [tok_to_id[ch] for ch in formula]
    if cls:
        ids = [tok_to_id["[CLS]"]] + ids
    real_len = len(ids)
    ids += [tok_to_id["[PAD]"]] * (max_len - real_len)
    mask = [1]*real_len + [0]*(max_len-real_len)
    return ids, mask

# model

class EncoderTransformer(nn.Module):
  def __init__(self):
    super().__init__()
    self.tok_emb = nn.Embedding(vocab_size, hidden, padding_idx=pad_id) #ver pq q tem hidden aqui, ver como isso funciona
    self.pos_emb = nn.Embedding(max_len, hidden) # de novo
    encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=hidden*4,batch_first=True) #o q eh batch first?
    self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=layers)
    self.mlm_head = nn.Linear(hidden, vocab_size) #ver como funcniona isso de ter uma MLM head

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

#dataset for encoding as goes (not doing it from the index, i need to change this)

class FormulaDataset(torch.utils.data.Dataset):
    def __init__(self, idx_list, max_len):
        self.idx_list = idx_list
        self.max_len = max_len

    def __len__(self):
        return len(self.idx_list)
    
    def __getitem__(self,idx):
        ids, mask = encode(self.idx_list[idx], self.max_len)
        return torch.tensor(ids), torch.tensor(mask)
    
# training

device = None
model = None
dataset = None
dataloader = None
dev_dataset = None
dev_dataloader = None
history = []
best_dev_loss = None
best_epoch = None

def eval_loss(dataloader):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch_input_ids, batch_attention_mask in dataloader:
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            masked_input, labels = mask_tokens(batch_input_ids, batch_attention_mask)
            _, logits = model(masked_input, batch_attention_mask)
            loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / n_batches


def main():
    print("starting")
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-name", type=str, required=True)
    parser.add_argument("--cls", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()

    global hidden, heads, layers, idx_t, dev_t, number_pl, cls, max_len, vocab, letters, tok_to_id, vocab_size, pad_id, mask_id

    folder =  Path(__file__).resolve().parent / "dataset" / args.folder_name

    cls = args.cls
    batch_size = args.batch_size
    epochs = args.epochs
    hidden = args.hidden
    heads = args.heads
    layers = args.layers


    # load the parameters the corpus was generated with
    act_world = pickle.load(open(folder / "act_world.pkl", "rb"))
    alt_worlds = pickle.load(open(folder / "alt_worlds.pkl", "rb"))
    number_pl, min_depth, max_depth, corpus_size, prop_td, n_worlds = pickle.load(open(folder / "params.pkl", "rb"))
    tfg.setup(number_pl_=number_pl, max_depth_=max_depth, act_world_=act_world, alt_worlds_=alt_worlds)
    
    # load the actual corpus
    idx_t = pickle.load(open(folder / "train.pkl", "rb"))
    dev_t = pickle.load(open(folder / "dev_t.pkl", "rb"))

    # build pieces for tokenization
    letters = list(string.ascii_lowercase)[:number_pl]
    vocab = (["[CLS]"] if cls else []) + ["[PAD]","[MASK]","∧","¬","(",")"," "] + letters
    tok_to_id = {tok: i for i, tok in enumerate(vocab)}
    max_len = max([len(str(tfg.true_le(i))) for i in idx_t])

    vocab_size = len(vocab)
    pad_id = tok_to_id["[PAD]"]
    mask_id = tok_to_id["[MASK]"]


    # create model, optimier and loss function
    global model, optimizer, loss_fn, dataloader, device, dataset, dev_dataset, dev_dataloader
    model = EncoderTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    #training
    probs = pickle.load(open(folder / "probs.pkl", "rb"))
    probs_t = torch.tensor(probs, dtype=torch.float)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("device:", device)

    dataset = FormulaDataset(idx_t, max_len)
    sampler = WeightedRandomSampler(weights=probs_t, num_samples=len(dataset), replacement=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    dev_dataset = FormulaDataset(dev_t, max_len)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)


    #os.makedirs("/kaggle/working/checkpoints", exist_ok=True)
    history = []
    best_dev_loss = float("inf")
    best_epoch = None

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch_input_ids, batch_attention_mask in dataloader:
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            loss = train_step(batch_input_ids, batch_attention_mask)
            total_loss += loss
            n_batches += 1
        train_loss = total_loss / n_batches
        dev_loss = eval_loss(dev_dataloader)
        history.append((epoch, train_loss, dev_loss))
        print(f"epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  dev_loss={dev_loss:.4f}")

        #torch.save(model.state_dict(), f"/kaggle/working/checkpoints/epoch_{epoch+1}.pt")

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_epoch = epoch + 1

    print(f"best epoch: {best_epoch}  best_dev_loss: {best_dev_loss:.4f}")

if __name__ == "__main__":
    main()


    #TODO: WHAT ABOUT THE ATTENTION MASK??
    #TODO: MISTURAR DEV_T E DEV_F??