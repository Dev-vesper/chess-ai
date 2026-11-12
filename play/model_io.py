"""بارگذاری مدل و گرفتن policy از MLP."""
import sys, os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import chess
from chessnet import make_mask, N_OUT
from build_dataset import encode, N_FEATURES


class PolicyNet:
    """Wrapper سبک برای model.npz بدون نیاز به re-init MLP."""
    def __init__(self, params):
        self.params = params

    def predict(self, X, mask):
        P = self.params
        z1 = X @ P["W1"] + P["b1"]
        h = np.maximum(z1, 0)
        logits = h @ P["W2"] + P["b2"]
        logits = np.where(mask, logits, -np.inf)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


def load_model(path="model.npz"):
    """model.npz رو بارگذاری می‌کنه. shape باید W1=(805, 128) باشه."""
    d = np.load(path)
    params = {k: d[k] for k in d.files}
    w1 = params["W1"]
    assert w1.shape[0] == N_FEATURES, \
        f"model W1 has {w1.shape[0]} inputs, expected {N_FEATURES}. "
    return PolicyNet(params)


def get_policy(net, board):
    """dict: real_move -> probability از دید مدل (از پرسپکتیو بازیکن جاری).

    مدل روی 'view' آموزش دیده: برای black، board.mirror() می‌شه.
    """
    if board.turn == chess.WHITE:
        view = board
        do_mirror = False
    else:
        view = board.mirror()
        do_mirror = True

    x = encode(view).astype(np.float32).reshape(1, -1)
    legal_view = list(view.legal_moves)
    if not legal_view:
        return {}

    L = np.array([[m.from_square * 64 + m.to_square for m in legal_view]],
                 dtype=np.int64)
    p = net.predict(x, make_mask(L))[0]

    policy = {}
    for mv in legal_view:
        prob = float(p[mv.from_square * 64 + mv.to_square])
        if do_mirror:
            real = chess.Move(chess.square_mirror(mv.from_square),
                              chess.square_mirror(mv.to_square),
                              mv.promotion)
        else:
            real = mv
        policy[real] = prob
    return policy
