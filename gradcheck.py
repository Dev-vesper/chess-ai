import numpy as np
from chessnet import MLP, make_mask
from build_dataset import N_FEATURES

d = np.load("dataset.npz")
n = 40
X = d["X"][:n].astype(np.float64)
y = d["Y"][:n].astype(np.int64)
mask = make_mask(d["L"][:n].astype(np.int64))

net = MLP(N_FEATURES, 16, seed=1, dtype=np.float64)
_, g = net.loss_and_grads(X, y, mask)

rng = np.random.default_rng(0)
eps, ok = 1e-4, True          # ← تغییر اصلی: 1e-5 → 1e-4
REL_TH = 1e-5

for k, W in net.params.items():
    flat = W.reshape(-1)
    gk = g[k].reshape(-1)
    nz = np.flatnonzero(gk != 0)       # همه‌ی گرادیان‌های غیرصفر
    worst = 0.0
    for i in rng.choice(nz, min(20, len(nz)), replace=False):
        old = flat[i]
        flat[i] = old + eps; lp, _ = net.loss_and_grads(X, y, mask)
        flat[i] = old - eps; lm, _ = net.loss_and_grads(X, y, mask)
        flat[i] = old
        num, ana = (lp - lm) / (2 * eps), gk[i]
        rel = abs(num - ana) / max(abs(num), abs(ana), 1e-8)
        worst = max(worst, rel)
    print(f"{k}: worst relative error = {worst:.2e}")
    ok &= worst < REL_TH
print("PASS" if ok else "FAIL")
