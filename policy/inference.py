"""Model loading and move-policy inference."""
import numpy as np

from policy.chessnet import MLP
from policy.encoding import (N_FEATURES, encode, legal_indices, make_mask,
                             map_move, move_index, view_for_turn)


def load_model(path="model.npz"):
    d = np.load(path)
    params = {k: d[k] for k in d.files}
    w1 = params["W1"]
    assert w1.shape[0] == N_FEATURES, \
        f"model W1 has {w1.shape[0]} inputs, expected {N_FEATURES}. "
    return MLP.from_params(params)


def get_policy(net, board):
    """dict: real_move -> probability from the side-to-move's perspective."""
    view, mirrored = view_for_turn(board)
    legal = list(view.legal_moves)
    if not legal:
        return {}

    x = encode(view).astype(np.float32).reshape(1, -1)
    L = np.array([legal_indices(view)], dtype=np.int64)
    p = net.predict(x, make_mask(L))[0]

    counts = {}
    for mv in legal:
        idx = move_index(mv)
        counts[idx] = counts.get(idx, 0) + 1
    return {map_move(mv, mirrored): float(p[move_index(mv)] / counts[move_index(mv)])
            for mv in legal}
