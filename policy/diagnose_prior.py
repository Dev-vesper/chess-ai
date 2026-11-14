"""Inspect model prior calibration on the starting position."""
import numpy as np
import chess

from policy.encoding import (encode, legal_indices, make_mask, move_from_index,
                             view_for_turn)
from policy.inference import load_model


def main():
    net = load_model()
    board = chess.Board()
    view, _ = view_for_turn(board)
    x = encode(view).astype(np.float32).reshape(1, -1)
    print(f"x shape: {x.shape}, sum: {x.sum()}, nonzero: {(x > 0).sum()}")

    legal = legal_indices(view)
    mask = make_mask(np.array([legal], dtype=np.int64))
    print(f"legal count: {len(legal)}, mask True: {mask.sum()}")

    p = net.predict(x, mask)[0]
    print(f"\np sum (all {p.size}): {p.sum():.6f}")
    print(f"p[legal] sum: {p[legal].sum():.6f}")
    print(f"p[legal] max: {p[legal].max():.4f}")
    print(f"p[legal] min: {p[legal].min():.6f}")
    print(f"p[legal] mean: {p[legal].mean():.4f}")

    print("\nTop 10 moves by prior:")
    for i in np.argsort(-p[legal])[:10]:
        mv = move_from_index(legal[i])
        print(f"  {mv.uci():6s}: {p[legal[i]]:.4f}")


if __name__ == "__main__":
    main()
