import numpy as np
stats = np.load("/home/isam/dev/MLX90640/python-inference/norm_stats.npz")
print("IR mean range:", stats["ir_mean"].min(), "to", stats["ir_mean"].max())
print("IR std range:",  stats["ir_std"].min(),  "to", stats["ir_std"].max())
print("Gas mean:", stats["gas_mean"])
print("Gas std:",  stats["gas_std"])
