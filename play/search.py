"""Alpha-Beta با prior تطبیقی (KL-regularized) + mate detection صریح."""
import sys, os, math
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import chess
from .evaluate import eval_negamax, MATE
from .model_io import get_policy

TEMPERATURE_TAU = 1.0
REPETITION_PENALTY = 30
MATE_THRESHOLD = MATE - 1000   # هر score ≥ 99,000 یعنی مات در درخت
LAM_CAP = 500.0                # سقف λ برای جلوگیری از فوران در موقعیت‌های مرزی


def quiescence(board, alpha, beta, ply, net, seen):
    if board.is_checkmate():
        return -MATE + ply
    if (board.is_stalemate() or board.is_insufficient_material()
            or board.can_claim_threefold_repetition()
            or board.can_claim_fifty_moves()):
        return 0

    key = board._transposition_key()
    # quiescence سریع: بدون mobility، بدون king safety
    stand = eval_negamax(board, full=False) - REPETITION_PENALTY * seen.get(key, 0)
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


def search_root(board, net, depth=3, top_k=8, tau=TEMPERATURE_TAU):
    """Alpha-Beta با mate detection + adaptive KL-regularized prior.

    Returns: (best_move, best_combined, info)
    info: list of (move, score, prior, combined, lam)
    """
    if board.is_game_over():
        return None, eval_negamax(board), []

    policy = get_policy(net, board)
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: policy.get(m, 0.0), reverse=True)
    if top_k:
        moves = moves[:top_k]
    if not moves:
        return None, eval_negamax(board), []

    seen = {board._transposition_key(): 1}
    raw_scores, priors = [], []

    for move in moves:
        board.push(move)
        k = board._transposition_key()
        seen[k] = seen.get(k, 0) + 1
        score = -alpha_beta(board, depth - 1, -MATE - 1, MATE + 1, 1, net, seen)
        seen[k] -= 1
        if seen[k] == 0:
            del seen[k]
        board.pop()
        raw_scores.append(score)
        priors.append(max(policy.get(move, 1e-9), 1e-12))

    scores_arr = np.array(raw_scores, dtype=np.float64)

    # ─── تشخیص مات: دسته‌ی جدا، prior نقشی ندارد ───
    if np.any(np.abs(scores_arr) >= MATE_THRESHOLD):
        best_idx = int(np.argmax(scores_arr))   # بهترین مات (یا کندترین باخت)
        best_move = moves[best_idx]
        best_combined = float(scores_arr[best_idx])
        info = [(m, s, priors[i], s, 0.0)
                for i, (m, s) in enumerate(zip(moves, raw_scores))]
        return best_move, best_combined, info

    # ─── حالت عادی: λ تطبیقی از داده‌ی همین موقعیت ───
    sigma = float(np.std(scores_arr)) if len(scores_arr) > 1 else 0.0
    lam = min(max(tau * sigma, 1.0), LAM_CAP)   # کف ۱، سقف ۵۰۰

    info = []
    best_combined = -MATE - 1.0
    best_move = None
    for move, score, prior in zip(moves, raw_scores, priors):
        log_bonus = lam * math.log(prior)
        combined = score + log_bonus
        info.append((move, score, prior, combined, lam))
        if combined > best_combined:
            best_combined = combined
            best_move = move

    return best_move, best_combined, info
