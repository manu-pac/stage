import argparse
import copy
import json
import math
import pickle

import numpy as np
import torch
import torch.nn as nn

FRACTIONS = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032,
             0.0625, 0.125, 0.25, 0.5, 1.0]


def build_probe(input_dim, n_classes, hidden=1000, n_hidden_layers=2):
    layers, d = [], input_dim
    for _ in range(n_hidden_layers):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers += [nn.Linear(d, n_classes)]
    return nn.Sequential(*layers)


@torch.no_grad()
def summed_nll_bits_and_acc(probe, X, y, batch_size, device):
    """Sum of -log2 p(y|x) over (X, y), plus accuracy."""
    probe.eval()
    ce_sum = nn.CrossEntropyLoss(reduction="sum")
    total_nats, correct = 0.0, 0
    for s in range(0, len(X), batch_size):
        xb = X[s:s + batch_size].to(device)
        yb = y[s:s + batch_size].to(device)
        logits = probe(xb)
        total_nats += ce_sum(logits, yb).item()
        correct += (logits.argmax(-1) == yb).sum().item()
    return total_nats / math.log(2), correct / len(X)


def train_probe(X_tr, y_tr, X_dev, y_dev, input_dim, n_classes, args, device):
    """Paper recipe: Adam lr 1e-3, halve lr on no-improvement epoch,
    stop after `patience` consecutive no-improvement epochs; return the
    best-dev-loss checkpoint."""
    torch.manual_seed(args.seed)
    probe = build_probe(input_dim, n_classes, args.hidden,
                        args.hidden_layers).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr)
    ce_mean = nn.CrossEntropyLoss()

    best_dev, best_state, bad_epochs = float("inf"), None, 0
    n = len(X_tr)
    for epoch in range(args.max_epochs):
        probe.train()
        perm = torch.randperm(n)
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            xb, yb = X_tr[idx].to(device), y_tr[idx].to(device)
            opt.zero_grad()
            loss = ce_mean(probe(xb), yb)
            loss.backward()
            opt.step()

        dev_bits, _ = summed_nll_bits_and_acc(probe, X_dev, y_dev,
                                              args.batch_size, device)
        if dev_bits < best_dev - 1e-7:
            best_dev, bad_epochs = dev_bits, 0
            best_state = copy.deepcopy(probe.state_dict())
        else:
            bad_epochs += 1
            for g in opt.param_groups:          # anneal lr by 0.5
                g["lr"] *= 0.5
            if bad_epochs >= args.patience:
                break

    probe.load_state_dict(best_state)
    return probe


def load_reps(path):
    """Load a .pkl file containing representations and return a float32
    numpy array of shape (N, H).

    Expected format (matches the extraction pipeline): a dict with a
    "reps" key holding an (N, H) ndarray, e.g.
    {"indexes": [...], "type": "t"/"f", "reps": ndarray (N, H)}.
    Also tolerates a bare ndarray/tensor, in case the format changes."""
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        if "reps" in obj:
            obj = obj["reps"]
        else:
            # fall back: look for a 2-D ndarray/tensor value among the dict's values
            for v in obj.values():
                if isinstance(v, (np.ndarray, torch.Tensor)) and getattr(v, "ndim", 0) == 2:
                    obj = v
                    break
            else:
                raise ValueError(
                    f"Could not find a 'reps' key or any 2-D array value in dict "
                    f"loaded from {path} (keys: {list(obj.keys())})")

    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
    elif isinstance(obj, np.ndarray):
        arr = obj
    else:
        raise ValueError(f"Unrecognized object of type {type(obj)} loaded from {path}")

    arr = np.asarray(arr, dtype=np.float32)
    assert arr.ndim == 2, (
        f"Expected (N, H) representations, got shape {arr.shape} from {path}. "
        f"Check that this pickle's 'reps' entry is per-example, not an already-averaged vector.")
    return arr


def build_dataset(true_path, false_path, dev_frac, seed):
    """Load true/false representation pickles, label them (1/0), merge,
    shuffle once with `seed`, and split off a dev set."""
    true_reps = load_reps(true_path)
    false_reps = load_reps(false_path)

    X = np.concatenate([true_reps, false_reps], axis=0)
    y = np.concatenate([np.ones(len(true_reps), dtype=np.int64),
                        np.zeros(len(false_reps), dtype=np.int64)], axis=0)

    print(f"loaded {len(true_reps)} true / {len(false_reps)} false examples, "
          f"dim={X.shape[1]}")

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(X))
    X, y = X[order], y[order]

    n_dev = max(1, int(round(dev_frac * len(X))))
    X_dev, y_dev = X[:n_dev], y[:n_dev]
    X_train, y_train = X[n_dev:], y[n_dev:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_dev = torch.tensor(X_dev, dtype=torch.float32)
    y_dev = torch.tensor(y_dev, dtype=torch.long)

    return X_train, y_train, X_dev, y_dev


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--true_path", required=True,
                   help=".pkl file with the true-class representations")
    p.add_argument("--false_path", required=True,
                   help=".pkl file with the false-class representations")
    p.add_argument("--dev_frac", type=float, default=0.1,
                   help="fraction of the combined true+false set held out as dev")
    p.add_argument("--out", default="online_code_results.json")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the train/dev split, shuffle, and probe init")
    p.add_argument("--hidden", type=int, default=1000)
    p.add_argument("--hidden_layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--min_first_block", type=int, default=50,
                   help="drop leading fractions until the first block has "
                        "at least this many examples")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train, X_dev, y_dev = build_dataset(
        args.true_path, args.false_path, args.dev_frac, args.seed)
    assert len(X_train) == len(y_train) and len(X_dev) == len(y_dev)

    n = len(X_train)
    d = X_train.shape[1]
    K = int(max(y_train.max(), y_dev.max()).item()) + 1
    print(f"{n} train / {len(X_dev)} dev examples, dim={d}, K={K} classes")

    # timesteps; drop leading fractions if first block too small
    fractions = list(FRACTIONS)
    while len(fractions) > 2 and int(fractions[0] * n) < args.min_first_block:
        fractions.pop(0)
    ts = sorted(set(max(1, int(f * n)) for f in fractions))
    ts[-1] = n
    print(f"timesteps (train-prefix sizes): {ts}")

    uniform_total = n * math.log2(K)
    codelength = ts[0] * math.log2(K)          # first block: uniform code
    portions = [{"prefix_size": ts[0], "note": "uniform code",
                 "block_size": ts[0], "block_bits": ts[0] * math.log2(K)}]

    for i in range(len(ts) - 1):
        prefix, nxt = ts[i], ts[i + 1]
        probe = train_probe(X_train[:prefix], y_train[:prefix],
                            X_dev, y_dev, d, K, args, device)
        block_bits, block_acc = summed_nll_bits_and_acc(
            probe, X_train[prefix:nxt], y_train[prefix:nxt],
            args.batch_size, device)
        _, dev_acc = summed_nll_bits_and_acc(probe, X_dev, y_dev,
                                             args.batch_size, device)
        codelength += block_bits
        portions.append({"prefix_size": prefix, "block_size": nxt - prefix,
                         "block_bits": block_bits, "block_acc": block_acc,
                         "dev_acc": dev_acc})
        print(f"trained on {prefix:>7d} -> block [{prefix},{nxt}): "
              f"{block_bits:10.1f} bits | dev acc {dev_acc:.4f}")

    # final model on all data: not part of MDL, only standard-probe accuracy
    probe = train_probe(X_train, y_train, X_dev, y_dev, d, K, args, device)
    _, final_dev_acc = summed_nll_bits_and_acc(probe, X_dev, y_dev,
                                               args.batch_size, device)

    results = {
        "n_train": n, "n_classes": K, "timesteps": ts,
        "codelength_bits": codelength,
        "codelength_kbits": codelength / 1024,
        "uniform_codelength_bits": uniform_total,
        "compression": uniform_total / codelength,
        "standard_probe_dev_acc": final_dev_acc,
        "portions": portions,
        "config": vars(args),
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nOnline codelength: {codelength / 1024:.2f} kbits")
    print(f"Uniform codelength: {uniform_total / 1024:.2f} kbits")
    print(f"Compression: {uniform_total / codelength:.2f}x")
    print(f"Standard probe (100% data) dev accuracy: {final_dev_acc:.4f}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()