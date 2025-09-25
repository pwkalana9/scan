import argparse, torch
import matplotlib.pyplot as plt

def show_az_range(cube, doppler, time):
    img = cube[:, :, doppler, time]
    plt.imshow(img, origin="lower", aspect="auto")
    plt.title(f"Azimuth-Range @ D={doppler}, T={time}")
    plt.xlabel("Range bin")
    plt.ylabel("Azimuth bin")
    plt.colorbar()
    plt.show()

def show_az_doppler(cube, rng, time):
    img = cube[:, rng, :, time]
    plt.imshow(img, origin="lower", aspect="auto")
    plt.title(f"Azimuth-Doppler @ R={rng}, T={time}")
    plt.xlabel("Doppler bin")
    plt.ylabel("Azimuth bin")
    plt.colorbar()
    plt.show()

def show_range_doppler(cube, az, time):
    img = cube[az, :, :, time]
    plt.imshow(img, origin="lower", aspect="auto")
    plt.title(f"Range-Doppler @ A={az}, T={time}")
    plt.xlabel("Doppler bin")
    plt.ylabel("Range bin")
    plt.colorbar()
    plt.show()

def show_doppler_time(cube, az, rng):
    img = cube[az, rng, :, :]
    plt.imshow(img, origin="lower", aspect="auto")
    plt.title(f"Doppler-Time @ A={az}, R={rng}")
    plt.xlabel("Time bin")
    plt.ylabel("Doppler bin")
    plt.colorbar()
    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, required=True)
    ap.add_argument("--stream", type=str, choices=["az_range","az_doppler","range_doppler","doppler_time"], required=True)
    ap.add_argument("--time", type=int, default=0)
    ap.add_argument("--doppler", type=int, default=0)
    ap.add_argument("--az", type=int, default=0)
    ap.add_argument("--rng", type=int, default=0)
    args = ap.parse_args()

    cube = torch.load(args.file).numpy()
    if args.stream == "az_range":
        show_az_range(cube, args.doppler, args.time)
    elif args.stream == "az_doppler":
        show_az_doppler(cube, args.rng, args.time)
    elif args.stream == "range_doppler":
        show_range_doppler(cube, args.az, args.time)
    elif args.stream == "doppler_time":
        show_doppler_time(cube, args.az, args.rng)

if __name__ == "__main__":
    main()

