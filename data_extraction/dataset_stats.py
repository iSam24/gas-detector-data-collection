import numpy as np, glob

for cls in ["normal", "aerosol", "flame", "breath"]:
    samples = glob.glob(f"dataset/{cls}/*.npz")
    if not samples:
        print(f"{cls}: no samples found")
        continue
    gas_all = np.array([np.load(s)["gas"] for s in samples])
    ir_all  = np.array([np.load(s)["ir_frames"] for s in samples])
    print(f"\n{cls} ({len(samples)} samples)")
    print(f"  Gas mean per channel : {gas_all.mean(axis=(0,1)).round(1)}")
    print(f"  Gas std  per channel : {gas_all.std(axis=(0,1)).round(1)}")
    print(f"  IR  mean             : {ir_all.mean():.2f}°C")
    print(f"  IR  std              : {ir_all.std():.2f}°C")
    