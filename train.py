# Trainingsskript
#
# Beispiel (laeuft auch auf CPU, ca. 15 min bei mir):
#   python train.py --n-layer 4 --n-head 4 --n-embd 256 --block-size 128 --max-iters 2000 --lr 2e-3

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from liollm.model import GPT, ModelConfig

parser = argparse.ArgumentParser()
# Modell
parser.add_argument("--block-size", type=int, default=256)
parser.add_argument("--n-layer", type=int, default=6)
parser.add_argument("--n-head", type=int, default=6)
parser.add_argument("--n-embd", type=int, default=384)
parser.add_argument("--dropout", type=float, default=0.1)
# Training
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--max-iters", type=int, default=5000)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--min-lr", type=float, default=1e-4)
parser.add_argument("--warmup-iters", type=int, default=200)
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--eval-interval", type=int, default=250)
parser.add_argument("--eval-iters", type=int, default=20)
parser.add_argument("--out-dir", type=str, default="checkpoints")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

torch.manual_seed(1337)
out_dir = Path(args.out_dir)
out_dir.mkdir(exist_ok=True)

# memmap, damit nicht alles in den RAM geladen werden muss (bei grossen Datensaetzen wichtig)
train_data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap("data/val.bin", dtype=np.uint16, mode="r")
vocab_size = 256 + len(json.loads(Path("data/tokenizer.json").read_text())["merges"])


def get_batch(split):
    data = train_data if split == "train" else val_data
    # zufaellige Startpositionen
    ix = torch.randint(len(data) - args.block_size - 1, (args.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + args.block_size].astype(np.int64)) for i in ix])
    # y ist x um eins verschoben -> das Modell soll immer das naechste Token vorhersagen
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + args.block_size].astype(np.int64)) for i in ix])
    return x.to(args.device), y.to(args.device)


def get_lr(it):
    # erst linear hochfahren (warmup), dann mit Cosinus runter auf min_lr
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters
    ratio = (it - args.warmup_iters) / (args.max_iters - args.warmup_iters)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * ratio))


@torch.no_grad()
def estimate_loss():
    # Loss ueber mehrere Batches mitteln, sonst ist der Wert zu verrauscht
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(args.eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


cfg = ModelConfig(
    vocab_size=vocab_size,
    block_size=args.block_size,
    n_layer=args.n_layer,
    n_head=args.n_head,
    n_embd=args.n_embd,
    dropout=args.dropout,
)
model = GPT(cfg).to(args.device)
print(f"Parameter: {model.num_params() / 1e6:.2f} Mio.")
optimizer = model.configure_optimizer(args.lr, args.weight_decay)

best_val = 1e9
history = []
start = time.time()

for it in range(args.max_iters + 1):
    lr = get_lr(it)
    for g in optimizer.param_groups:
        g["lr"] = lr

    if it % args.eval_interval == 0:
        losses = estimate_loss()
        history.append({"iter": it, **losses})
        print(f"iter {it}: train {losses['train']:.4f}, val {losses['val']:.4f}, "
              f"zeit {time.time() - start:.0f}s")
        # nur speichern wenn es besser geworden ist (wegen Overfitting)
        if losses["val"] < best_val:
            best_val = losses["val"]
            torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, out_dir / "ckpt.pt")
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    if it == args.max_iters:
        break

    x, y = get_batch("train")
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gegen zu grosse Gradienten
    optimizer.step()

print("bester val loss:", round(best_val, 4))
