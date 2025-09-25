import argparse, json, random
from pathlib import Path

def build_split(data_dir: str, out_dir: str, split=(0.7,0.2,0.1), seed=42):
    random.seed(seed)
    scenes = sorted([p for p in Path(data_dir).glob("scene_*") if (p/"ardt.pt").exists()])
    random.shuffle(scenes)
    n = len(scenes)
    n_train = int(split[0]*n)
    n_val   = int(split[1]*n)
    train = scenes[:n_train]
    val   = scenes[n_train:n_train+n_val]
    test  = scenes[n_train+n_val:]

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out/"splits").mkdir(exist_ok=True, parents=True)

    def write_index(name, arr):
        idx = out/"splits"/f"{name}.txt"
        idx.write_text("\n".join(str(p.resolve()) for p in arr))
        return idx

    ti = write_index("train", train)
    vi = write_index("val", val)
    te = write_index("test", test)

    meta = dict(total=n, train=len(train), val=len(val), test=len(test), split=split)
    (out/"meta.json").write_text(json.dumps(meta, indent=2))
    return str(ti), str(vi), str(te)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--split", nargs=3, type=float, default=[0.7,0.2,0.1])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build_split(args.data, args.out, tuple(args.split), seed=args.seed)

if __name__ == "__main__":
    main()

