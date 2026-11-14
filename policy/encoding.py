"""Board and move encoding contract shared by dataset, training, and inference."""
import numpy as np
import chess

N_FEATURES = 805   # 768 piece + 4 castling + 32 tactical + 1 en passant
N_OUT = 4096       # from_square * 64 + to_square

PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
             chess.ROOK: 5, chess.QUEEN: 9}
HANG_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
CENTER = [chess.E4, chess.D4, chess.E5, chess.D5]


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

    # --- en passant ---
    x[804] = int(board.ep_square is not None)

    return x


def make_mask(L):
    """L: (B, maxlen) legal move indices padded with -1 -> boolean mask (B, N_OUT)."""
    mask = np.zeros((L.shape[0], N_OUT), dtype=bool)
    r, c = np.nonzero(L >= 0)
    mask[r, L[r, c]] = True
    return mask


def move_index(move):
    return move.from_square * 64 + move.to_square


def move_from_index(idx):
    return chess.Move(idx // 64, idx % 64)


def legal_indices(board):
    return sorted({move_index(m) for m in board.legal_moves})


def view_for_turn(board):
    """Position from the side-to-move's perspective (Black is mirrored)."""
    if board.turn == chess.WHITE:
        return board, False
    return board.mirror(), True


def map_move(move, mirrored):
    """Map a move between the real board and the model view (mirroring is an involution)."""
    if not mirrored:
        return move
    return chess.Move(chess.square_mirror(move.from_square),
                      chess.square_mirror(move.to_square),
                      move.promotion)
