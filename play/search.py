"""Alpha-Beta با prior وزن‌دهی‌شده به‌صورت تطبیقی (KL-regularized).

فرمول ترکیب:
    σ = std({score(a) : a ∈ candidates})
    λ = τ · σ
    combined(a) = score(a) + λ · log(prior(a))

λ از پراکندگی امتیازهای جستجو در همان موقعیت محاسبه می‌شود، نه از
یک وزن ثابت. این همان ایده‌ی KL-regularized policy optimization است:
    π ∝ prior · exp(Q / λ)
که در آن λ نقش دما را بازی می‌کند. انتخاب λ = τ·σ به این معناست که
«یک انحراف معیار اختلاف در امتیاز، معادل e برابر اختلاف در prior است»
— یک انتخاب بی‌بعد که با داده‌ی هر موقعیت خودش را تنظیم می‌کند.
"""
import sys, os, math
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import chess
from .evaluate import eval_negamax, MATE
from .model_io import get_policy

# τ دمای بی‌بعد KL regularization. ۱.۰ = مقیاس طبیعی.
TEMPERATURE_TAU = 1.0
# جریمه‌ی تکرار برای جلوگیری از shuffle.
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


def search_root(board, net, depth=3, top_k=8, tau=TEMPERATURE_TAU):
    """Alpha-Beta با prior وزن‌دهی‌شده به‌صورت تطبیقی.

    Returns:
        (best_move, best_combined, info)
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

    # full-window در root: به همه‌ی امتیازها نیاز داریم تا σ را حساب کنیم.
    # (بدون alpha-beta pruning در root — کمی کندتر، ولی تنها راه درست.)
    seen = {board._transposition_key(): 1}
    raw_scores = []
    priors = []

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

    # ─── محاسبه‌ی λ تطبیقی از داده‌ی همین موقعیت ───
    scores_arr = np.array(raw_scores, dtype=np.float64)
    sigma = float(np.std(scores_arr)) if len(scores_arr) > 1 else 0.0
    lam = max(tau * sigma, 1.0)  # کف ۱ cp برای جلوگیری از lam=0

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
