"""Alpha-Beta با model prior در eval برگ‌ها + quiescence."""
import sys, os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import chess
from .evaluate import eval_negamax, MATE
from .model_io import get_policy

MODEL_WEIGHT = 80
REPETITION_PENALTY = 30


def quiescence(board, alpha, beta, ply, net, seen):
    if board.is_checkmate():
        return -MATE + ply
    if (board.is_stalemate() or board.is_insufficient_material()
            or board.can_claim_threefold_repetition()
            or board.can_claim_fifty_moves()):
        return 0

    key = board._transposition_key()
    stand = eval_negamax(board) - REPETITION_PENALTY * seen.get(key, 0)
    if stand >= beta:
        return beta
    if stand > alpha:
        alpha = stand

    in_check = board.is_check()
    if in_check and ply > 6:
        return alpha

    for move in board.legal_moves:
        if not in_check and not (board.is_capture(move) or move.promotion):
            continue
        board.push(move)
        k = board._transposition_key()
        seen[k] = seen.get(k, 0) + 1
        score = -quiescence(board, -beta, -alpha, ply + 1, net, seen)
        seen[k] -= 1
        if seen[k] == 0:
            del seen[k]
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def alpha_beta(board, depth, alpha, beta, ply, net, seen):
    if board.is_checkmate():
        return -MATE + ply
    if (board.is_stalemate() or board.is_insufficient_material()
            or board.can_claim_threefold_repetition()
            or board.can_claim_fifty_moves()):
        return 0
    if depth == 0:
        return quiescence(board, alpha, beta, ply, net, seen)

    moves = list(board.legal_moves)
    moves.sort(key=lambda m: (board.is_capture(m), m.promotion is not None),
               reverse=True)

    for move in moves:
        board.push(move)
        k = board._transposition_key()
        seen[k] = seen.get(k, 0) + 1
        score = -alpha_beta(board, depth - 1, -beta, -alpha, ply + 1, net, seen)
        seen[k] -= 1
        if seen[k] == 0:
            del seen[k]
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def search_root(board, net, depth=3, top_k=8):
    """بهترین حرکت با model ordering + model bonus + alpha-beta."""
    if board.is_game_over():
        return None, eval_negamax(board), []

    policy = get_policy(net, board)
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: policy.get(m, 0.0), reverse=True)
    if top_k:
        moves = moves[:top_k]

    n_legal = len(policy)
    seen = {board._transposition_key(): 1}
    best_score = -MATE - 1
    best_move = None
    alpha = -MATE - 1
    info = []

    for move in moves:
        board.push(move)
        k = board._transposition_key()
        seen[k] = seen.get(k, 0) + 1
        score = -alpha_beta(board, depth - 1, -MATE - 1, -alpha, 1, net, seen)
        seen[k] -= 1
        if seen[k] == 0:
            del seen[k]
        board.pop()

        prior = policy.get(move, 0.0)
        if n_legal > 1:
            confidence = (prior - 1.0 / n_legal) / (1.0 - 1.0 / n_legal)
        else:
            confidence = 0.0
        combined = score + int(MODEL_WEIGHT * confidence * 0.5)

        info.append((move, score, prior, combined))
        if combined > best_score:
            best_score = combined
            best_move = move
            alpha = max(alpha, score)

    return best_move, best_score, info
