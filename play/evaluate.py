"""ارزیابی ایستا با material + PST + hanging pieces + king safety."""
import chess

MATE = 100_000

PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20_000,
}

# PST از دید سفید. برای سیاه با square_mirror.
PST = {
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


def eval_white(board):
    """نمره از دید سفید (+ = سفید بهتر)."""
    score = 0
    for pt, base in PIECE_VALUES.items():
        table = PST.get(pt)
        for sq in board.pieces(pt, chess.WHITE):
            score += base + (table[sq] if table else 0)
        for sq in board.pieces(pt, chess.BLACK):
            score -= base + (table[chess.square_mirror(sq)] if table else 0)
    return score


def _hanging_penalty(board):
    """مجموع ارزش مهره‌های hanging (از دید سفید)."""
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


def eval_negamax(board):
    """نمره از دید بازیکن جاری (+ = بهتر)."""
    if board.is_checkmate():
        return -MATE
    if (board.is_stalemate() or board.is_insufficient_material()
            or board.is_seventyfive_moves() or board.is_fivefold_repetition()
            or board.can_claim_threefold_repetition()
            or board.can_claim_fifty_moves()):
        return 0

    s = eval_white(board)
    s += _hanging_penalty(board)

    if board.is_check():
        s += 30 if board.turn == chess.WHITE else -30

    return s if board.turn == chess.WHITE else -s
