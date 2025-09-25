import argparse, torch
from pathlib import Path
from torch.utils.data import DataLoader
from .datasets.base import ARDT4DDataset
from .models.pvt4d import PVT4D

@torch.no_grad()
def run(model, loader, device):
    model.eval()
    all_stats = []
    for x,y in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits)
        pred = (prob>0.5).float()
        i = (pred*y).sum()
        u = (pred+y - pred*y).sum()
        iou = (i/(u+1e-6)).item()
        all_stats.append(iou)
    return sum(all_stats)/len(all_stats) if all_stats else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--t_frames", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    te_idx = Path(args.data)/"splits"/"test.txt"
    test_set = ARDT4DDataset(str(te_idx))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = PVT4D(t_frames=args.t_frames).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    mean_iou = run(model, test_loader, device)
    print(f"Test mIoU: {mean_iou:.4f}")

if __name__ == "__main__":
    main()

