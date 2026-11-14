"""Terminal and draw detection shared by evaluation, search, and play."""
import chess

MATE = 100_000


def is_claimable_draw(board):
    return (board.can_claim_threefold_repetition()
            or board.can_claim_fifty_moves())


def is_search_draw(board):
    return (board.is_stalemate() or board.is_insufficient_material()
            or is_claimable_draw(board))


def is_eval_draw(board):
    return (is_search_draw(board) or board.is_seventyfive_moves()
            or board.is_fivefold_repetition())
