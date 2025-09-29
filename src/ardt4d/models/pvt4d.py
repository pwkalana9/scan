import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3DPatchEmbed(nn.Module):
    '''Fold T into channels, then Conv3d over (A,R,D).'''
    def __init__(self, in_ch=1, embed_dim=64, t_frames=10):
        super().__init__()
        self.t_frames = t_frames
        self.proj = nn.Conv3d(in_ch*t_frames, embed_dim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        # x: (B,1,A,R,D,T)
        B,C,A,R,D,T = x.shape
        if T != self.t_frames:
            raise ValueError(f"Expected T={self.t_frames}, got {T}")
        x = x.reshape(B, C*T, A, R, D)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim),
        )

    def forward(self, x):
        # x: (B,N,C)
        h = x
        x = self.norm1(x)
        x,_ = self.attn(x,x,x, need_weights=False)
        x = x + h
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + h
        return x

class PVT4D(nn.Module):
    def __init__(self, t_frames=10, in_ch=1, widths=(64,128,256), depth=(2,2,4), heads=(2,4,8)):
        super().__init__()
        self.t_frames = t_frames
        self.stage0 = Conv3DPatchEmbed(in_ch, widths[0], t_frames)
        self.down1  = nn.Conv3d(widths[0], widths[1], kernel_size=3, stride=2, padding=1)
        self.down2  = nn.Conv3d(widths[1], widths[2], kernel_size=3, stride=2, padding=1)

        def enc(w,d,h): return nn.ModuleList([TransformerBlock(w, num_heads=h) for _ in range(d)])
        self.enc0 = enc(widths[0], depth[0], heads[0])
        self.enc1 = enc(widths[1], depth[1], heads[1])
        self.enc2 = enc(widths[2], depth[2], heads[2])

        self.up1   = nn.ConvTranspose3d(widths[2], widths[1], kernel_size=2, stride=2)
        self.up0   = nn.ConvTranspose3d(widths[1], widths[0], kernel_size=2, stride=2)
        self.fuse1 = nn.Conv3d(widths[1]*2, widths[1], 1)
        self.fuse0 = nn.Conv3d(widths[0]*2, widths[0], 1)
        self.head  = nn.Conv3d(widths[0], 1, 1)

    def _to_tokens(self, x):
        B,C,A,R,D = x.shape
        x = x.view(B, C, A*R*D).transpose(1,2)  # (B,N,C)
        return x, (A,R,D)

    def _from_tokens(self, x, shp):
        A,R,D = shp
        B,N,C = x.shape
        x = x.transpose(1,2).view(B, C, A, R, D)
        return x

    def forward(self, x):
        # x: (B,1,A,R,D,T)
        x0 = self.stage0(x)
        t0, shp0 = self._to_tokens(x0)
        for blk in self.enc0: t0 = blk(t0)
        x0 = self._from_tokens(t0, shp0)

        x1 = self.down1(x0)
        t1, shp1 = self._to_tokens(x1)
        for blk in self.enc1: t1 = blk(t1)
        x1 = self._from_tokens(t1, shp1)

        x2 = self.down2(x1)
        t2, shp2 = self._to_tokens(x2)
        for blk in self.enc2: t2 = blk(t2)
        x2 = self._from_tokens(t2, shp2)

        u1 = self.up1(x2)
        if u1.shape[2:] != x1.shape[2:]:
            u1 = F.interpolate(u1, size=x1.shape[2:], mode='trilinear', align_corners=False)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.fuse1(u1)

        u0 = self.up0(u1)
        if u0.shape[2:] != x0.shape[2:]:
            u0 = F.interpolate(u0, size=x0.shape[2:], mode='trilinear', align_corners=False)
        u0 = torch.cat([u0, x0], dim=1)
        u0 = self.fuse0(u0)

        logits = self.head(u0)  # (B,1,A,R,D)
        logits = logits.unsqueeze(-1).repeat(1,1,1,1,1,self.t_frames)
        return logits
