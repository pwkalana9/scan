import argparse, math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .datasets.base import ARDT4DDataset
# Use the improved time-specific model
from .models.pvt4d_v2 import PVT4D_V2

# ---------------- Losses ----------------
def bce_loss_weighted(pred, target, pos_weight=20.0):
    # pred: logits; target: 0/1
    pw = torch.tensor(pos_weight, device=pred.device, dtype=pred.dtype)
    return F.binary_cross_entropy_with_logits(pred, target, pos_weight=pw)

def focal_tversky_loss(pred, target, alpha=0.7, beta=0.3, gamma=0.75, eps=1e-6):
    p = torch.sigmoid(pred)
    tp = (p*target).sum()
    fp = (p*(1-target)).sum()
    fn = ((1-p)*target).sum()
    tversky = (tp + eps) / (tp + alpha*fp + beta*fn + eps)
    return (1.0 - tversky).pow(gamma)

@torch.no_grad()
def dice_score(pred, target, eps=1e-6):
    p = (torch.sigmoid(pred) > 0.5).float()
    num = 2*(p*target).sum()
    den = (p.sum() + target.sum() + eps)
    return (num/den).item()

# --------------- Train / Eval ---------------
def train_epoch(model, loader, opt, scaler, sched, device, pos_weight, alpha, beta, gamma):
    model.train()
    running = 0.0
    for x,y in loader:
        x = x.to(device)  # (B,1,A,R,D,T)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(x)
            loss = 0.4*bce_loss_weighted(logits, y, pos_weight=pos_weight) + \
                   0.6*focal_tversky_loss(logits, y, alpha=alpha, beta=beta, gamma=gamma)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if sched is not None:
            sched.step()
        running += loss.item()*x.size(0)
    return running/len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, device, pos_weight, alpha, beta, gamma):
    model.eval()
    running = 0.0
    dsc = 0.0
    n = 0
    for x,y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = 0.4*bce_loss_weighted(logits, y, pos_weight=pos_weight) + \
               0.6*focal_tversky_loss(logits, y, alpha=alpha, beta=beta, gamma=gamma)
        running += loss.item()*x.size(0)
        dsc += dice_score(logits, y)*x.size(0)
        n += x.size(0)
    return running/len(loader.dataset), (dsc/max(1,n))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path with splits/train.txt, val.txt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--t_frames", type=int, default=10)
    ap.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    ap.add_argument("--num_workers", type=int, default=0)
    # loss hyperparams
    ap.add_argument("--pos_weight", type=float, default=20.0)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.75)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_idx = Path(args.data)/"splits"/"train.txt"
    va_idx = Path(args.data)/"splits"/"val.txt"

    train_set = ARDT4DDataset(str(tr_idx))
    val_set   = ARDT4DDataset(str(va_idx))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = PVT4D_V2(t_frames=args.t_frames).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Cosine schedule with warmup (per-step)
    total_steps = args.epochs * max(1, len(train_loader))
    warmup = max(1, 5 * len(train_loader)) if total_steps > 5*len(train_loader) else max(1, total_steps//10)
    def lr_lambda(step):
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best = 1e9
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    for ep in range(1, args.epochs+1):
        tr = train_epoch(model, train_loader, opt, scaler, sched, device,
                         pos_weight=args.pos_weight, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
        va, dsc = eval_epoch(model, val_loader, device,
                             pos_weight=args.pos_weight, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
        print(f"[{ep:03d}] train_loss={tr:.4f}  val_loss={va:.4f}  val_dice={dsc:.4f}  lr={sched.get_last_lr()[0]:.3e}")

        torch.save({"model":model.state_dict(),"epoch":ep}, ckpt_dir/"last.pt")
        if va < best:
            best = va
            torch.save({"model":model.state_dict(),"epoch":ep}, ckpt_dir/"best.pt")

    print("Done. Best val loss:", best)

if __name__ == "__main__":
    main()
