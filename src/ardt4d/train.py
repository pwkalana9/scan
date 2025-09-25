import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .datasets.base import ARDT4DDataset
from .models.pvt4d import PVT4D

def dice_loss(pred, target, eps=1e-6):
    prob = torch.sigmoid(pred)
    dims = list(range(1, pred.ndim))
    num = 2*(prob*target).sum(dim=dims)
    den = (prob*prob + target*target).sum(dim=dims) + eps
    return 1 - (num/den).mean()

def bce_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)

def train_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    for x,y in loader:
        x = x.to(device); y = y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = 0.5*bce_loss(logits, y) + 0.5*dice_loss(logits, y)
        loss.backward()
        opt.step()
        total += loss.item()*x.size(0)
    return total/len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = 0.0
    for x,y in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss = 0.5*bce_loss(logits, y) + 0.5*dice_loss(logits, y)
        total += loss.item()*x.size(0)
    return total/len(loader.dataset)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--t_frames", type=int, default=10)
    ap.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_idx = Path(args.data)/"splits"/"train.txt"
    va_idx = Path(args.data)/"splits"/"val.txt"
    train_set = ARDT4DDataset(str(tr_idx))
    val_set   = ARDT4DDataset(str(va_idx))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = PVT4D(t_frames=args.t_frames).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = 1e9
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs+1):
        tr = train_epoch(model, train_loader, opt, device)
        va = eval_epoch(model, val_loader, device)
        print(f"[{ep:03d}] train={tr:.4f} val={va:.4f}")
        torch.save({"model":model.state_dict(),"epoch":ep}, ckpt_dir/"last.pt")
        if va < best:
            best = va
            torch.save({"model":model.state_dict(),"epoch":ep}, ckpt_dir/"best.pt")
    print("Done. Best val loss:", best)

if __name__ == "__main__":
    main()

