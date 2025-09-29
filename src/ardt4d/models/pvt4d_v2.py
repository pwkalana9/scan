import torch
import torch.nn as nn
import torch.nn.functional as F
from .pvt4d import TransformerBlock  # reuse the block

class SpatialUNetPVT(nn.Module):
    """A UNet-like encoder/decoder over (A,R,D) that returns (B,1,A,R,D)."""
    def __init__(self, in_ch=1, widths=(64,128,256), depth=(2,2,4), heads=(2,4,8)):
        super().__init__()
        self.stem = nn.Conv3d(in_ch, widths[0], 3, padding=1, stride=1)

        def enc(w,d,h): return nn.ModuleList([TransformerBlock(w, num_heads=h) for _ in range(d)])
        self.enc0 = enc(widths[0], depth[0], heads[0])
        self.down1 = nn.Conv3d(widths[0], widths[1], 3, stride=2, padding=1)
        self.enc1 = enc(widths[1], depth[1], heads[1])
        self.down2 = nn.Conv3d(widths[1], widths[2], 3, stride=2, padding=1)
        self.enc2 = enc(widths[2], depth[2], heads[2])

        self.up1   = nn.ConvTranspose3d(widths[2], widths[1], 2, stride=2)
        self.fuse1 = nn.Conv3d(widths[1]*2, widths[1], 1)
        self.up0   = nn.ConvTranspose3d(widths[1], widths[0], 2, stride=2)
        self.fuse0 = nn.Conv3d(widths[0]*2, widths[0], 1)
        self.head  = nn.Conv3d(widths[0], 1, 1)

    def _tok(self, x):
        B,C,A,R,D = x.shape
        x = x.view(B, C, A*R*D).transpose(1,2)
        return x,(A,R,D)

    def _untok(self, x, shp):
        A,R,D = shp
        B,N,C = x.shape
        return x.transpose(1,2).view(B, C, A, R, D)

    def forward(self, x):  # (B,1,A,R,D)
        x0 = self.stem(x)
        t0, shp0 = self._tok(x0)
        for blk in self.enc0: t0 = blk(t0)
        x0 = self._untok(t0, shp0)

        x1 = self.down1(x0)
        t1, shp1 = self._tok(x1)
        for blk in self.enc1: t1 = blk(t1)
        x1 = self._untok(t1, shp1)

        x2 = self.down2(x1)
        t2, shp2 = self._tok(x2)
        for blk in self.enc2: t2 = blk(t2)
        x2 = self._untok(t2, shp2)

        u1 = self.up1(x2)
        if u1.shape[2:] != x1.shape[2:]:
            u1 = F.interpolate(u1, size=x1.shape[2:], mode="trilinear", align_corners=False)
        u1 = self.fuse1(torch.cat([u1, x1], dim=1))

        u0 = self.up0(u1)
        if u0.shape[2:] != x0.shape[2:]:
            u0 = F.interpolate(u0, size=x0.shape[2:], mode="trilinear", align_corners=False)
        u0 = self.fuse0(torch.cat([u0, x0], dim=1))

        return self.head(u0)  # (B,1,A,R,D)

class PVT4D_V2(nn.Module):
    """
    Frame-wise spatial encoder-decoder with shared weights → time-specific logits.
    Input:  (B,1,A,R,D,T)
    Output: (B,1,A,R,D,T)
    """
    def __init__(self, t_frames=10, in_ch=1, widths=(64,128,256), depth=(2,2,4), heads=(2,4,8)):
        super().__init__()
        self.t_frames = t_frames
        self.spatial = SpatialUNetPVT(in_ch=in_ch, widths=widths, depth=depth, heads=heads)

    def forward(self, x):
        B,C,A,R,D,T = x.shape
        if T != self.t_frames:
            raise ValueError(f"Expected T={self.t_frames}, got {T}")
        xt = x.permute(0,5,1,2,3,4).contiguous().view(B*T, C, A, R, D)  # (B*T,1,A,R,D)
        logits = self.spatial(xt)                                       # (B*T,1,A,R,D)
        logits = logits.view(B, T, 1, A, R, D).permute(0,2,3,4,5,1)     # (B,1,A,R,D,T)
        return logits
