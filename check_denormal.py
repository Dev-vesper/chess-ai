import numpy as np
m = np.load("model.npz")
tiny = np.finfo(np.float32).tiny   # کوچک‌ترین عدد نرمال float32 ≈ 1.18e-38
for k in m.files:
    w = m[k]
    nz = w != 0
    den = (np.abs(w) < tiny) & nz
    print(f"{k}: denormal count = {den.sum()} / {w.size}")
