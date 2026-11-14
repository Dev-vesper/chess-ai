import argparse
import numpy as np
import chess, chess.pgn, chess.polyglot

from dataset.pgn import pgn_files
from policy.encoding import (encode, legal_indices, map_move, move_index,
                             view_for_turn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--name", nargs="+", required=True)
    ap.add_argument("--out", default="dataset.npz")
    a = ap.parse_args()
    mine = {n.lower() for n in a.name}

    X, Y, G, P, LEG, POS_KEY = [], [], [], [], [], []
    used = skipped = promos = not_legal = 0
    game_id = 0

    for fp in pgn_files(a.pgn):
        with open(fp, encoding="utf-8", errors="replace") as f:
            while True:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                h = g.headers
                if h.get("Variant", "Standard") not in ("Standard", "From Position") or g.errors:
                    skipped += 1; continue
                w, b = h.get("White", "").lower(), h.get("Black", "").lower()
                if w in mine: color = chess.WHITE
                elif b in mine: color = chess.BLACK
                else:
                    skipped += 1; continue

                board = g.board()
                for move in g.mainline_moves():
                    if board.turn == color:
                        view, mirrored = view_for_turn(board)
                        target = move_index(map_move(move, mirrored))
                        if move.promotion and move.promotion != chess.QUEEN:
                            promos += 1
                        legal = legal_indices(view)
                        if target not in legal:
                            not_legal += 1
                        X.append(encode(view)); Y.append(target)
                        G.append(game_id); P.append(board.fullmove_number)
                        LEG.append(legal)
                        # کلید یکتای موقعیت: Zobrist hash (سازگار با Polyglot)
                        POS_KEY.append(chess.polyglot.zobrist_hash(view))
                    board.push(move)
                game_id += 1
                used += 1

    maxlen = max(len(l) for l in LEG)
    L = np.full((len(LEG), maxlen), -1, dtype=np.int16)
    for i, l in enumerate(LEG):
        L[i, :len(l)] = l

    POS_KEY = np.array(POS_KEY, dtype=np.uint64)
    np.savez_compressed(a.out, X=np.array(X, np.uint8), Y=np.array(Y, np.int16),
                        G=np.array(G, np.int32), P=np.array(P, np.int16), L=L,
                        POS_KEY=POS_KEY)
    print(f"games used: {used} | skipped: {skipped}")
    print(f"samples: {len(Y)} | unique targets: {len(set(Y))} | under-promotions: {promos}")
    print(f"target NOT in legal set (must be 0): {not_legal} | max legal moves: {maxlen}")
    print(f"unique positions (by Zobrist): {len(np.unique(POS_KEY))}")


if __name__ == "__main__":
    main()
