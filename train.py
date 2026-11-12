import time, argparse
import numpy as np
from chessnet import MLP, Adam, make_mask
from build_dataset import N_FEATURES

ap = argparse.ArgumentParser()
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--drop", type=float, default=0.5)
ap.add_argument("--wd", type=float, default=1e-3)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--patience", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--val_ratio", type=float, default=0.15,
                help="نسبت موقعیت‌های val (بر اساس هش موقعیت)")
ap.add_argument("--flush", type=int, default=1)
a = ap.parse_args()

TINY = np.finfo(np.float32).tiny
FLUSH_TH = 1e-30

d = np.load("dataset.npz")
X = d["X"].astype(np.float32); Y = d["Y"].astype(np.int64)
G = d["G"]; L = d["L"].astype(np.int64); PLY = d["P"]
POS_KEY = d["POS_KEY"]

# ---------- تقسیم بر اساس هش موقعیت (بدون نشت) ----------
# hash(position) % 100 < val_ratio*100  →  val
# بقیه  →  train
val_thresh = int(a.val_ratio * 100)
is_val = (POS_KEY % np.uint64(100)) < np.uint64(val_thresh)

# حذف قطعی: هر موقعیتی که در val است، از train حذف می‌شود
val_keys = set(POS_KEY[is_val].tolist())
train_mask = ~is_val
for k in val_keys:
    train_mask &= (POS_KEY != k)
tr = np.flatnonzero(train_mask)
va = np.flatnonzero(is_val)

# گزارش نشت (باید صفر باشد)
train_keys = set(POS_KEY[tr].tolist())
leak = len(val_keys & train_keys)
print(f"train: {len(tr)} samples | val: {len(va)} samples")
print(f"unique val positions: {len(val_keys)} | leakage (must be 0): {leak}")

def evaluate(net, idx):
    loss = top1 = top3 = 0.0
    early = np.zeros(2); late = np.zeros(2)
    for s in range(0, len(idx), 512):
        b = idx[s:s + 512]
        mask = make_mask(L[b])
        p = net.predict(X[b], mask)
        y = Y[b]
        loss += -np.log(p[np.arange(len(b)), y] + 1e-12).sum()
        hit1 = p.argmax(1) == y
        hit3 = (np.argpartition(-p, 2, axis=1)[:, :3] == y[:, None]).any(1)
        top1 += hit1.sum(); top3 += hit3.sum()
        e = PLY[b] <= 10
        early += [hit1[e].sum(), e.sum()]
        late += [hit1[~e].sum(), (~e).sum()]
    n = len(idx)
    return loss / n, top1 / n, top3 / n, early[0] / max(early[1], 1), late[0] / max(late[1], 1)

def count_denormals(state):
    return sum(int(((np.abs(v) < TINY) & (v != 0)).sum()) for v in state.values())

def flush_small(state):
    for v in state.values():
        v[np.abs(v) < FLUSH_TH] = 0.0

# ---------- baselines ----------
mask_va = make_mask(L[va])
n_legal = mask_va.sum(1)
base_rand_acc = np.mean(1.0 / n_legal)
base_rand_loss = np.mean(np.log(n_legal))
counts = np.bincount(Y[tr], minlength=4096)
pred_freq = np.where(mask_va, counts, -1).argmax(1)
base_freq_acc = np.mean(pred_freq == Y[va])
e = PLY[va] <= 10
print(f"baseline random-legal : top1 {base_rand_acc:.3f} | loss {base_rand_loss:.3f}")
print(f"baseline most-frequent: top1 {base_freq_acc:.3f} "
      f"(opening {np.mean((pred_freq==Y[va])[e]):.3f}, later {np.mean((pred_freq==Y[va])[~e]):.3f})")
print(f"flush denormals: {'ON' if a.flush else 'OFF'}\n")

# ---------- آموزش ----------
net = MLP(N_FEATURES, a.hidden, seed=a.seed)
opt = Adam(net.params, lr=a.lr)

best_vl, best_params, bad = 1e9, None, 0   # early stop بر اساس val loss (نه top1)

for ep in range(1, a.epochs + 1):
    t0 = time.time()
    order = rng.permutation(tr) if (rng := np.random.default_rng(a.seed + ep)) else tr
    tot = 0.0; nb = 0
    for s in range(0, len(order), a.batch):
        b = order[s:s + a.batch]
        loss, g = net.loss_and_grads(X[b], Y[b], make_mask(L[b]),
                                     drop=a.drop, rng=rng, wd=a.wd)
        opt.step(net.params, g)
        if a.flush:
            flush_small(opt.m); flush_small(opt.v)
        tot += loss; nb += 1
    t_train = time.time() - t0

    vl, v1, v3, _, _ = evaluate(net, va)
    den_m, den_v = count_denormals(opt.m), count_denormals(opt.v)
    print(f"ep {ep:3d} | train {tot/nb:.3f} | val {vl:.3f} | "
          f"val top1 {v1:.3f} top3 {v3:.3f} | {t_train:.1f}s "
          f"| denormals m:{den_m} v:{den_v}")

    if vl < best_vl:
        best_vl, best_params, bad = vl, {k: v.copy() for k, v in net.params.items()}, 0
    else:
        bad += 1
        if bad >= a.patience:
            print("early stop"); break

net.params.update(best_params)
tl, t1, t3, _, _ = evaluate(net, tr)
vl, v1, v3, ve, vlate = evaluate(net, va)
print("\n=== نتیجه‌ی بهترین مدل (بر اساس val loss) ===")
print(f"train: loss {tl:.3f} top1 {t1:.3f} top3 {t3:.3f}")
print(f"val  : loss {vl:.3f} top1 {v1:.3f} top3 {v3:.3f}  "
      f"(opening ≤10: {ve:.3f} | later: {vlate:.3f})")
np.savez("model.npz", **net.params)
