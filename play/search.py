"""Alpha-Beta با move ordering از مدل + quiescence."""
import sys, os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import chess
from .evaluate import eval_negamax, MATE
from .model_io import get_policy


def quiescence(board, alpha, beta, ply):
    """جستجوی فقط capture/promotion تا آرامش تاکتیکی."""
    if board.is_checkmate():
        return -MATE + ply
    stand = eval_negamax(board)
    if stand >= beta:
        return beta
    if stand > alpha:
        alpha = stand

    in_check = board.is_check()
    if in_check and ply > 6:  # محدودیت عمق در check chain
        return alpha

    # در check: همه‌ی حرکت‌ها، وگرنه فقط capture/promotion
    for move in board.legal_moves:
        if not in_check and not (board.is_capture(move) or move.promotion):
            continue
        board.push(move)
        score = -quiescence(board, -beta, -alpha, ply + 1)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def alpha_beta(board, depth, alpha, beta, ply=0):
    if board.is_checkmate():
        return -MATE + ply
    if (board.is_stalemate() or board.is_insufficient_material()
            or board.is_seventyfive_moves() or board.is_fivefold_repetition()):
        return 0
    if depth == 0:
        return quiescence(board, alpha, beta, ply)

    # ordering ساده: capture و promotion اول
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: (board.is_capture(m), m.promotion is not None),
               reverse=True)

    for move in moves:
        board.push(move)
        score = -alpha_beta(board, depth - 1, -beta, -alpha, ply + 1)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def search_root(board, net, depth=3, top_k=8):
    """بهترین حرکت در root با model ordering + alpha-beta.

    Returns: (best_move, best_score, top_moves_info)
    """
    if board.is_game_over():
        return None, eval_negamax(board), []

    policy = get_policy(net, board)
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: policy.get(m, 0.0), reverse=True)
    if top_k:
        moves = moves[:top_k]

    best_score = -MATE - 1
    best_move = None
    alpha = -MATE - 1
    info = []

    for move in moves:
        board.push(move)
        score = -alpha_beta(board, depth - 1, -MATE - 1, -alpha, ply=1)
        board.pop()
        info.append((move, score, policy.get(move, 0.0)))
        if score > best_score:
            best_score = score
            best_move = move
            alpha = max(alpha, score)
    return best_move, best_score, info
