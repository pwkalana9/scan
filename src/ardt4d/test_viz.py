import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .datasets.base import ARDT4DDataset
from .models.pvt4d_v2 import PVT4D_V2  # works with either v2 or v1 if you swap

def _unravel_index(flat_idx: torch.Tensor, shape):
    a, r = shape
    i = flat_idx // r
    j = flat_idx % r
    return int(i), int(j)

@torch.no_grad()
def visualize_sample(x, y, prob, save_path: Path,
                     az_idx=None, rng_idx=None, d_idx=None, t_idx=None,
                     threshold=0.5, show=False):
    A, R, D, T = y.shape
    y_sum_AR = y.sum(dim=(2, 3)) or prob.sum(dim=(2, 3))  # (A,R)
    y_sum_DT = y.sum(dim=(0, 1)) or prob.sum(dim=(0, 1))  # (D,T)

    if az_idx is None or rng_idx is None:
        flat = torch.argmax(y_sum_AR.view(-1))
        az_idx, rng_idx = _unravel_index(flat, (A, R))
    if d_idx is None or t_idx is None:
        flat = torch.argmax(y_sum_DT.view(-1))
        d_idx, t_idx = _unravel_index(flat, (D, T))

    gt_DT   = y[az_idx, rng_idx, :, :]
    pr_DT_p = prob[az_idx, rng_idx, :, :]
    pr_DT_b = (pr_DT_p > threshold).float()

    gt_AR   = y[:, :, d_idx, t_idx]
    pr_AR_p = prob[:, :, d_idx, t_idx]
    pr_AR_b = (pr_AR_p > threshold).float()

    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(2, 2, 1); im1 = ax1.imshow(gt_DT.cpu().numpy(), origin="lower", aspect="auto")
    ax1.set_title(f"GT: Doppler–Time @ A={az_idx}, R={rng_idx}"); ax1.set_xlabel("Time"); ax1.set_ylabel("Doppler"); fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax2 = fig.add_subplot(2, 2, 2); im2 = ax2.imshow(pr_DT_p.cpu().numpy(), origin="lower", aspect="auto")
    ax2.set_title(f"Pred: Doppler–Time (prob)"); ax2.set_xlabel("Time"); ax2.set_ylabel("Doppler"); fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax3 = fig.add_subplot(2, 2, 3); im3 = ax3.imshow(gt_AR.cpu().numpy(), origin="lower", aspect="auto")
    ax3.set_title(f"GT: Azimuth–Range @ D={d_idx}, T={t_idx}"); ax3.set_xlabel("Range"); ax3.set_ylabel("Azimuth"); fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax4 = fig.add_subplot(2, 2, 4); im4 = ax4.imshow(pr_AR_p.cpu().numpy(), origin="lower", aspect="auto")
    ax4.set_title(f"Pred: Azimuth–Range (prob)"); ax4.set_xlabel("Range"); ax4.set_ylabel("Azimuth"); fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    fig.suptitle("ARDT Detection vs Ground Truth — DT & AR views", y=0.98, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)

    fig2 = plt.figure(figsize=(12, 5))
    axb1 = fig2.add_subplot(1, 2, 1); axb1.imshow(pr_DT_b.cpu().numpy(), origin="lower", aspect="auto"); axb1.set_title("Pred DT (binary)")
    axb2 = fig2.add_subplot(1, 2, 2); axb2.imshow(pr_AR_b.cpu().numpy(), origin="lower", aspect="auto"); axb2.set_title("Pred AR (binary)")
    fig2.tight_layout(); fig2.savefig(save_path.with_name(save_path.stem + "_bin.png"), dpi=150); plt.close(fig2)

@torch.no_grad()
def run_and_visualize(data_dir: Path, ckpt_path: Path, t_frames=10, batch_size=1, threshold=0.5,
                      az=None, rng=None, d=None, t=None, save_dir: Path=Path("./viz_out"),
                      max_samples=5, device=None, show=False):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_idx = Path(data_dir) / "splits" / "test.txt"
    ds = ARDT4DDataset(str(test_idx))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model = PVT4D_V2(t_frames=t_frames).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    processed = 0
    for bidx, (x, y) in enumerate(dl):
        x = x.to(device); y = y.to(device)
        logits = model(x); prob = torch.sigmoid(logits)
        B = x.shape[0]
        for i in range(B):
            y_i = y[i, 0]; p_i = prob[i, 0]
            out_file = Path(save_dir) / f"sample_{bidx:03d}_{i:02d}.png"
            visualize_sample(None, y_i, p_i, out_file, az_idx=az, rng_idx=rng, d_idx=d, t_idx=t,
                             threshold=threshold, show=show)
            processed += 1
            if processed >= max_samples:
                return

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--t_frames", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--save_dir", type=str, default="./viz_out")
    ap.add_argument("--max_samples", type=int, default=5)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--dt_az", type=int, default=None)
    ap.add_argument("--dt_rng", type=int, default=None)
    ap.add_argument("--ar_d", type=int, default=None)
    ap.add_argument("--ar_t", type=int, default=None)
    args = ap.parse_args()

    run_and_visualize(Path(args.data), Path(args.ckpt), t_frames=args.t_frames, batch_size=args.batch_size,
                      threshold=args.threshold, az=args.dt_az, rng=args.dt_rng, d=args.ar_d, t=args.ar_t,
                      save_dir=Path(args.save_dir), max_samples=args.max_samples, show=args.show)

if __name__ == "__main__":
    main()
