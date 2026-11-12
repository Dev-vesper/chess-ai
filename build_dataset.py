import glob, argparse, hashlib
import numpy as np
import chess, chess.pgn, chess.polyglot

N_FEATURES = 805   # 768 piece + 4 castling + 32 tactical + 1 en passant

PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
             chess.ROOK: 5, chess.QUEEN: 9}
HANG_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
CENTER = [chess.E4, chess.D4, chess.E5, chess.D5]

def _stable_hash(s: str) -> int:
    """هش پایدار (blake2b) — برخلاف hash() پایتون، بین اجراها ثابت است."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")

def _hanging(board, sq, attacker_color):
    owner = board.color_at(sq)
    if owner is None:
        return False
    return (board.is_attacked_by(attacker_color, sq)
            and not board.is_attacked_by(owner, sq))

def encode(board):
    x = np.zeros(N_FEATURES, dtype=np.uint8)

    # --- 768 بیت piece ---
    for sq, p in board.piece_map().items():
        idx = (p.piece_type - 1) + (0 if p.color == chess.WHITE else 6)
        x[idx * 64 + sq] = 1

    # --- 4 بیت castling ---
    x[768] = board.has_kingside_castling_rights(chess.WHITE)
    x[769] = board.has_queenside_castling_rights(chess.WHITE)
    x[770] = board.has_kingside_castling_rights(chess.BLACK)
    x[771] = board.has_queenside_castling_rights(chess.BLACK)

    me, opp = board.turn, not board.turn

    # --- is_check ---
    x[772] = int(board.is_check())

    # --- مهره‌های من که hang هستن ---
    for i, pt in enumerate(HANG_TYPES):
        if any(_hanging(board, sq, opp) for sq in board.pieces(pt, me)):
            x[773 + i] = 1

    # --- مهره‌های حریف که می‌تونم بزنم ---
    for i, pt in enumerate(HANG_TYPES):
        if any(_hanging(board, sq, me) for sq in board.pieces(pt, opp)):
            x[778 + i] = 1

    # --- تعادل مادی ---
    def mat(color):
        return sum(PIECE_VAL[pt] * len(board.pieces(pt, color)) for pt in PIECE_VAL)
    diff = max(-5, min(5, mat(me) - mat(opp)))
    x[783 + diff + 5] = 1          # 783..793

    # --- کنترل مرکز ---
    my_ctrl  = sum(board.is_attacked_by(me,  s) for s in CENTER)
    opp_ctrl = sum(board.is_attacked_by(opp, s) for s in CENTER)
    x[794 + my_ctrl]  = 1          # 794..798
    x[799 + opp_ctrl] = 1          # 799..803

    # --- en passant (جدید) ---
    x[804] = int(board.ep_square is not None)

    return x

def pgn_files(path):
    return [path] if path.endswith(".pgn") else sorted(glob.glob(path.rstrip("/") + "/*.pgn"))

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
                        if color == chess.BLACK:
                            view = board.mirror()
                            fr = chess.square_mirror(move.from_square)
                            to = chess.square_mirror(move.to_square)
                        else:
                            view, fr, to = board, move.from_square, move.to_square
                        if move.promotion and move.promotion != chess.QUEEN:
                            promos += 1
                        legal = sorted({m.from_square * 64 + m.to_square for m in view.legal_moves})
                        target = fr * 64 + to
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
                        POS_KEY=POS_KEY)   # ← اضافه شد
    print(f"games used: {used} | skipped: {skipped}")
    print(f"samples: {len(Y)} | unique targets: {len(set(Y))} | under-promotions: {promos}")
    print(f"target NOT in legal set (must be 0): {not_legal} | max legal moves: {maxlen}")
    print(f"unique positions (by Zobrist): {len(np.unique(POS_KEY))}")

if __name__ == "__main__":
    main()
