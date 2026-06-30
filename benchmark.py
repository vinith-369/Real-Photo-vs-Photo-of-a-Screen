import time
import os
import glob
from predict import predict

real_paths = glob.glob("data/Real/*.*")[:10]

total_time = 0
for path in real_paths:
    t0 = time.time()
    predict(path)
    t1 = time.time()
    total_time += (t1 - t0)

avg_latency = (total_time / len(real_paths)) * 1000
print(f"Average latency: {avg_latency:.2f} ms")
