from pathlib import Path
import torch
from torch.utils.data import Dataset

class ARDT4DDataset(Dataset):
    def __init__(self, index_file: str):
        super().__init__()
        self.items = [p.strip() for p in Path(index_file).read_text().splitlines() if p.strip()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sdir = Path(self.items[i])
        cube = torch.load(sdir/"ardt.pt").float()  # (A,R,D,T)
        mask = torch.load(sdir/"mask.pt").float()  # (A,R,D,T)
        x = cube.unsqueeze(0)  # (1,A,R,D,T)
        y = mask.unsqueeze(0)  # (1,A,R,D,T)
        return x, y

