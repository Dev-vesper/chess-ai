import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import chess
from build_dataset import encode, N_FEATURES
from chessnet import make_mask

d = np.load("model.npz")
params = {k: d[k] for k in d.files}

board = chess.Board()
x = encode(board).astype(np.float32).reshape(1, -1)
print(f"x shape: {x.shape}, sum: {x.sum()}, nonzero: {(x > 0).sum()}")

legal = [m.from_square * 64 + m.to_square for m in board.legal_moves]
L = np.array([legal], dtype=np.int64)
mask = make_mask(L)
print(f"legal count: {len(legal)}, mask True: {mask.sum()}")

z1 = x @ params["W1"] + params["b1"]
h = np.maximum(z1, 0)
logits = h @ params["W2"] + params["b2"]

masked = np.where(mask, logits, -np.inf)
e = np.exp(masked - masked.max(axis=1, keepdims=True))
p = e / e.sum(axis=1, keepdims=True)

print(f"\np sum (all 4096): {p.sum():.6f}")
print(f"p[legal] sum: {p[0, legal].sum():.6f}")
print(f"p[legal] max: {p[0, legal].max():.4f}")
print(f"p[legal] min: {p[0, legal].min():.6f}")
print(f"p[legal] mean: {p[0, legal].mean():.4f}")

print("\nTop 10 moves by prior:")
idx = np.argsort(-p[0, legal])[:10]
for i in idx:
    sq = legal[i]
    mv = chess.Move(sq // 64, sq % 64)
    print(f"  {mv.uci():6s}: {p[0, legal[i]]:.4f}")
