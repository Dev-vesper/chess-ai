"""Eval با دو فاز (mid/end) + mobility سریع + hanging + king safety."""
import chess

MATE = 100_000

PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20_000,
}

PST_MID = {
    chess.PAWN: [
          0,  0,  0,  0,  0,  0,  0,  0,
         50, 50, 50, 50, 50, 50, 50, 50,
         10, 10, 20, 30, 30, 20, 10, 10,
          5,  5, 10, 25, 25, 10,  5,  5,
          0,  0,  0, 20, 20,  0,  0,  0,
          5, -5,-10,  0,  0,-10, -5,  5,
          5, 10, 10,-20,-20, 10, 10,  5,
          0,  0,  0,  0,  0,  0,  0,  0,
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    chess.ROOK: [
          0,  0,  0,  0,  0,  0,  0,  0,
          5, 10, 10, 10, 10, 10, 10,  5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
          0,  0,  0,  5,  5,  0,  0,  0,
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20,
    ],
    chess.KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20,
    ],
}

PST_END = {**{k: v for k, v in PST_MID.items() if k != chess.KING},  # بقیه همان mid
    chess.KING: [
        -50,-40,-30,-20,-20,-30,-40,-50,
        -40,-30,-20,  0,  0,-20,-30,-40,
        -30,-20,  0, 20, 20,  0,-20,-30,
        -20,  0, 20, 40, 40, 20,  0,-20,
        -20,  0, 20, 40, 40, 20,  0,-20,
        -30,-20,  0, 20, 20,  0,-20,-30,
        -40,-30,-20,  0,  0,-20,-30,-40,
        -50,-40,-30,-20,-20,-30,-40,-50,
    ],
}


def _phase(board):
    """0 = endgame, 24 = fully midgame."""
    p = 0
    for pt, v in [(chess.KNIGHT, 1), (chess.BISHOP, 1),
                  (chess.ROOK, 2), (chess.QUEEN, 4)]:
        p += v * (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK)))
    return min(p, 24)


def _mobility(board):
    """شبه‌قانونی، بدون اعتبارسنجی — ~۵–۱۰ برابر سریع‌تر از legal_moves."""
    saved = board.turn
    board.turn = chess.WHITE
    w = sum(1 for _ in board.pseudo_legal_moves)
    board.turn = chess.BLACK
    b = sum(1 for _ in board.pseudo_legal_moves)
    board.turn = saved
    return w - b


def _hanging_penalty(board):
    s = 0
    for pt, base in PIECE_VALUES.items():
        if pt == chess.KING:
            continue
        for sq in board.pieces(pt, chess.WHITE):
            if (board.is_attacked_by(chess.BLACK, sq)
                    and not board.is_attacked_by(chess.WHITE, sq)):
                s -= base // 3
        for sq in board.pieces(pt, chess.BLACK):
            if (board.is_attacked_by(chess.WHITE, sq)
                    and not board.is_attacked_by(chess.BLACK, sq)):
                s += base // 3
    return s


def _king_safety(board, color):
    ksq = board.king(color)
    if ksq is None:
        return 0
    file = chess.square_file(ksq)
    rank = chess.square_rank(ksq)
    d = 1 if color == chess.WHITE else -1
    shield = 0
    for df in (-1, 0, 1):
        f, r = file + df, rank + d
        if 0 <= f < 8 and 0 <= r < 8:
            p = board.piece_at(chess.square(f, r))
            if p is not None and p.piece_type == chess.PAWN and p.color == color:
                shield += 10
    return shield


def eval_white(board, full=True):
    """full=True: mid/end PST + mobility + hanging + king safety.
       full=False: فقط mid PST + hanging (برای quiescence)."""
    phase = _phase(board) if full else 24
    mid_w = phase / 24
    end_w = 1 - mid_w

    score = 0
    for pt, base in PIECE_VALUES.items():
        for sq in board.pieces(pt, chess.WHITE):
            score += base + int(mid_w * PST_MID[pt][sq] + end_w * PST_END[pt][sq])
        for sq in board.pieces(pt, chess.BLACK):
            m = chess.square_mirror(sq)
            score -= base + int(mid_w * PST_MID[pt][m] + end_w * PST_END[pt][m])

    score += _hanging_penalty(board)

    if full:
        score += 2 * _mobility(board)
        score += int(mid_w * (_king_safety(board, chess.WHITE) - _king_safety(board, chess.BLACK)))

    return score


def eval_negamax(board, full=True):
    if board.is_checkmate():
        return -MATE
    if (board.is_stalemate() or board.is_insufficient_material()
            or board.is_seventyfive_moves() or board.is_fivefold_repetition()
            or board.can_claim_threefold_repetition()
            or board.can_claim_fifty_moves()):
        return 0
    s = eval_white(board, full=full)
    if board.is_check():
        s += 30 if board.turn == chess.WHITE else -30
    return s if board.turn == chess.WHITE else -s
