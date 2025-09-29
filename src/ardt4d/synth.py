import os, json, math, random, argparse
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np
import torch
from .config import RadarDims, RadarPhys, Shadows

def latlon_to_local_xy(lat0: float, lon0: float, lat: float, lon: float) -> Tuple[float,float]:
    R_earth = 6371000.0
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    latm = math.radians(lat0)
    x = R_earth * dlon * math.cos(latm)
    y = R_earth * dlat
    return x, y

def velocity_to_doppler_bin(vr_mps: float, D: int, v_max_mps: float) -> int:
    vr_clip = max(-v_max_mps, min(v_max_mps, vr_mps))
    frac = (vr_clip + v_max_mps) / (2 * v_max_mps + 1e-9)
    return int(frac * (D - 1))

def pos_to_az_range(x: float, y: float, A: int, R: int, az_fov_deg: float, r_max: float):
    az = math.degrees(math.atan2(y, x))
    az_min = -az_fov_deg/2
    az_max = +az_fov_deg/2
    if az < az_min or az > az_max:
        return None, None
    az_frac = (az - az_min) / (az_max - az_min + 1e-9)
    az_bin = int(az_frac * (A - 1))
    rng = math.hypot(x, y)
    if rng < 0 or rng > r_max:
        return None, None
    r_bin = int((rng / (r_max + 1e-9)) * (R - 1))
    return az_bin, r_bin

def gaussian_add(cube: np.ndarray, center, sigmas, amp: float):
    A,R,D,T = cube.shape
    ca, cr, cd, ct = center
    sa, sr, sd, st = sigmas
    a = np.arange(A)[:,None,None,None]
    r = np.arange(R)[None,:,None,None]
    d = np.arange(D)[None,None,:,None]
    t = np.arange(T)[None,None,None,:]
    ga = np.exp(-0.5*((a - ca)/(sa+1e-9))**2)
    gr = np.exp(-0.5*((r - cr)/(sr+1e-9))**2)
    gd = np.exp(-0.5*((d - cd)/(sd+1e-9))**2)
    gt = np.exp(-0.5*((t - ct)/(st+1e-9))**2)
    cube += (amp * ga*gr*gd*gt).astype(cube.dtype)

def stamp_mask(mask: np.ndarray, center, radii=(1,2,1,0), val=1):
    """Write a small cuboid around the center to reduce extreme sparsity."""
    A,R,D,T = mask.shape
    ca,cr,cd,ct = center
    ra,rr,rd,rt = radii
    a0,a1 = max(0, ca-ra), min(A-1, ca+ra)
    r0,r1 = max(0, cr-rr), min(R-1, cr+rr)
    d0,d1 = max(0, cd-rd), min(D-1, cd+rd)
    t0,t1 = max(0, ct-rt), min(T-1, ct+rt)
    mask[a0:a1+1, r0:r1+1, d0:d1+1, t0:t1+1] = val

def simulate_scene(dims: RadarDims, phys: RadarPhys, sh: Shadows,
                   n_objects: int = 3,
                   seed: Optional[int] = None,
                   gps_tracks: Optional[List[Dict]] = None):
    if seed is not None:
        np.random.seed(seed); random.seed(seed)

    A,R,D,T = dims.A, dims.R, dims.D, dims.T
    cube = np.zeros((A,R,D,T), dtype=np.float32)
    mask = np.zeros((A,R,D,T), dtype=np.uint8)
    objects = []

    tracks_used = 0
    if gps_tracks:
        for gt in gps_tracks:
            lat0, lon0 = gt["lat0"], gt["lon0"]
            track = gt["track"]  # (t, lat, lon)
            times = [p[0] for p in track]
            tmin, tmax = min(times), max(times)
            for k in range(T):
                tk = tmin + (tmax - tmin) * (k/(T-1 if T>1 else 1))
                idx = int(np.argmin([abs(t - tk) for t in times]))
                lat, lon = track[idx][1], track[idx][2]
                x,y = latlon_to_local_xy(lat0, lon0, lat, lon)
                j = min(idx+1, len(track)-1)
                lat2, lon2 = track[j][1], track[j][2]
                x2,y2 = latlon_to_local_xy(lat0, lon0, lat2, lon2)
                dt = max(1e-3, track[j][0]-track[idx][0])
                vx, vy = (x2-x)/dt, (y2-y)/dt

                az_bin, r_bin = pos_to_az_range(x, y, A, R, phys.az_fov_deg, phys.r_max)
                if az_bin is None: continue
                rnorm = math.hypot(x,y) + 1e-6
                ux, uy = x/rnorm, y/rnorm
                vr = vx*ux + vy*uy
                d_bin = velocity_to_doppler_bin(vr, D, phys.v_max_mps)

                gaussian_add(cube, (az_bin, r_bin, d_bin, k), (0.8,1.2,1.0,0.3), amp=3.0)
                stamp_mask(mask, (az_bin, r_bin, d_bin, k))
                for s in range(1,5):
                    w = math.exp(-s/sh.decay_az)
                    azb = min(A-1, max(0, az_bin+s))
                    gaussian_add(cube, (azb, r_bin, d_bin, k), (1.2,1.4,1.1,0.4), amp=1.5*w)
                for s in range(1,5):
                    w = math.exp(-s/sh.decay_r)
                    rb = min(R-1, max(0, r_bin+s))
                    gaussian_add(cube, (az_bin, rb, d_bin, k), (1.0,1.8,1.1,0.4), amp=1.2*w)

            objects.append({"type":"gps", "points":len(track)})
            tracks_used += 1

    for _ in range(max(0, n_objects - tracks_used)):
        rng0 = random.uniform(200.0, phys.r_max*0.8)
        az0 = random.uniform(-phys.az_fov_deg/2*math.pi/180.0, phys.az_fov_deg/2*math.pi/180.0)
        x0, y0 = rng0*math.cos(az0), rng0*math.sin(az0)
        speed = random.uniform(0.0, phys.v_max_mps*0.8)
        heading = random.uniform(-math.pi, math.pi)
        vx, vy = speed*math.cos(heading), speed*math.sin(heading)
        ax, ay = random.uniform(-1.0,1.0), random.uniform(-1.0,1.0)

        for k in range(T):
            t = float(k)
            x = x0 + vx*t + 0.5*ax*t*t
            y = y0 + vy*t + 0.5*ay*t*t
            vxk = vx + ax*t
            vyk = vy + ay*t
            arz, rb = pos_to_az_range(x, y, A, R, phys.az_fov_deg, phys.r_max)
            if arz is None: continue
            rnorm = math.hypot(x,y) + 1e-6
            ux, uy = x/rnorm, y/rnorm
            vr = vxk*ux + vyk*uy
            db = velocity_to_doppler_bin(vr, D, phys.v_max_mps)

            gaussian_add(cube, (arz, rb, db, k), (0.8,1.2,1.0,0.3), amp=3.2)
            stamp_mask(mask, (arz, rb, db, k))
            for s in range(1,5):
                w = math.exp(-s/sh.decay_az)
                azb = min(A-1, max(0, arz+s))
                gaussian_add(cube, (azb, rb, db, k), (1.2,1.4,1.1,0.4), amp=1.4*w)
            for s in range(1,5):
                w = math.exp(-s/sh.decay_r)
                rb2 = min(R-1, max(0, rb+s))
                gaussian_add(cube, (arz, rb2, db, k), (1.0,1.8,1.1,0.4), amp=1.1*w)

        objects.append({"type":"kinematic_random"})

    # noise floor + clip
    noise = np.random.gamma(shape=1.5, scale=0.5, size=cube.shape).astype(np.float32) * 0.2
    cube += noise
    cube = np.clip(cube, 0.0, None)

    # robust per-scene normalization (99th percentile)
    p99 = float(np.percentile(cube, 99.0))
    scale = max(p99, 1e-6)
    cube = cube / scale

    meta = dict(A=A,R=R,D=D,T=T, n_objects=n_objects,
                lambda_m=phys.lambda_m, v_max_mps=phys.v_max_mps,
                az_fov_deg=phys.az_fov_deg, r_max=phys.r_max,
                shadow_decay_az=sh.decay_az, shadow_decay_r=sh.decay_r, objects=objects)
    return cube, mask, meta

def save_scene(out_dir: Path, idx: int, cube, mask, meta):
    sdir = out_dir / f"scene_{idx:03d}"
    sdir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(cube), sdir/"ardt.pt")
    torch.save(torch.from_numpy(mask), sdir/"mask.pt")
    with open(sdir/"meta.json","w") as f:
        json.dump(meta, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n_scenes", type=int, default=20)
    ap.add_argument("--A", type=int, default=22)
    ap.add_argument("--R", type=int, default=100)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gps_csv", type=str, default=None)
    args = ap.parse_args()

    dims = RadarDims(args.A, args.R, args.D, args.T)
    phys = RadarPhys()
    sh = Shadows()
    random.seed(args.seed); np.random.seed(args.seed)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    gps_tracks = None
    if args.gps_csv and os.path.exists(args.gps_csv):
        import csv
        track = []
        with open(args.gps_csv,"r") as f:
            reader = csv.reader(f)
            for r in reader:
                try:
                    t, lat, lon = float(r[0]), float(r[1]), float(r[2])
                    track.append((t, lat, lon))
                except:
                    continue
        if track:
            lat0, lon0 = track[0][1], track[0][2]
            gps_tracks = [dict(lat0=lat0, lon0=lon0, track=track)]

    for i in range(args.n_scenes):
        cube, mask, meta = simulate_scene(dims, phys, sh, gps_tracks=gps_tracks)
        save_scene(out, i, cube, mask, meta)

if __name__ == "__main__":
    main()
