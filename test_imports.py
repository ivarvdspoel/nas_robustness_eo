import time
start = time.time()
import torch
total = time.time() - start

print("Import time: ", total)


device = "cuda"
size = 1024 * 1024 * 256  # ~1 GB (float32)

x = torch.randn(size, dtype=torch.float32)

# Warmup
for _ in range(3):
    y = x.to(device)
    z = y.to("cpu")

# Measure CPU → GPU
start = time.time()
y = x.to(device)
torch.cuda.synchronize()
t1 = time.time() - start

# Measure GPU → CPU
start = time.time()
z = y.to("cpu")
torch.cuda.synchronize()
t2 = time.time() - start

gb = x.numel() * 4 / 1e9
print(f"CPU → GPU: {gb/t1:.2f} GB/s")
print(f"GPU → CPU: {gb/t2:.2f} GB/s")