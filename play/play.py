"""CLI برای بازی با مدل + Alpha-Beta."""
import sys, os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import argparse
import chess
from play.model_io import load_model
from play.search import search_root


def print_board(board):
    print()
    print(board)
    print(f"FEN: {board.fen()}")
    print()


def is_draw_claimable(board):
    return (board.can_claim_threefold_repetition()
            or board.can_claim_fifty_moves())


def pick_human_move(board):
    while True:
        s = input("Your move (SAN or UCI, e.g. 'e4' or 'e2e4'): ").strip()
        if s.lower() in ("quit", "exit"):
            return None
        try:
            try:
                move = board.parse_san(s)
            except ValueError:
                move = chess.Move.from_uci(s)
            if move in board.legal_moves:
                return move
            print(f"Illegal: {move.uci()}")
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError) as e:
            print(f"Invalid: {e}")


def play_game(net, human_color, depth, top_k, max_plies=300):
    board = chess.Board()
    print_board(board)

    for ply in range(max_plies):
        if board.is_game_over() or is_draw_claimable(board):
            if board.is_game_over():
                reason = board.outcome().termination.name
            else:
                reason = "claimable draw (3-fold/50-move)"
            print(f"Game over: {board.result()} ({reason})")
            return board

        if board.turn == human_color:
            move = pick_human_move(board)
            if move is None:
                print("Quit.")
                return board
        else:
            print(f"AI thinking (depth={depth}, top_k={top_k})...")
            move, score, info = search_root(board, net, depth=depth, top_k=top_k)
            if move is None:
                print(f"No legal move. score={score}")
                return board
            info_sorted = sorted(info, key=lambda x: -x[3])[:3]
            print(f"AI plays: {board.san(move)} | eval={score} "
                  f"| top3: {[(m.uci(), round(s), round(p,3)) for m,s,p,_ in info_sorted]}")

        board.push(move)
        print_board(board)

    return board


def self_play(net, depth, top_k, games=1):
    for g in range(games):
        print(f"\n=== Self-play game {g+1}/{games} ===")
        board = chess.Board()
        for ply in range(300):
            if board.is_game_over() or is_draw_claimable(board):
                if board.is_game_over():
                    reason = board.outcome().termination.name
                else:
                    reason = "claimable draw (3-fold/50-move)"
                print(f"Result: {board.result()} ({reason})")
                print(f"Plys: {ply}")
                break
            move, score, _ = search_root(board, net, depth=depth, top_k=top_k)
            if move is None:
                break
            print(f"{ply+1:3d}. {board.san(move):8s}  eval={score:+6d}")
            board.push(move)
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.npz")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--human", choices=["white", "black"], default=None)
    ap.add_argument("--self-play", action="store_true")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--fen", default=None)
    a = ap.parse_args()

    net = load_model(a.model)
    print(f"Loaded model: W1={net.params['W1'].shape}, "
          f"W2={net.params['W2'].shape}")

    if a.fen:
        board = chess.Board(a.fen)
        print_board(board)
        move, score, info = search_root(board, net, depth=a.depth, top_k=a.top_k)
        print(f"Best: {board.san(move) if move else 'None'} | eval={score}")
        for m, s, p, c in sorted(info, key=lambda x: -x[3])[:5]:
            print(f"  {m.uci():6s} score={s:+6d} prior={p:.3f} combined={c:+6d}")
        return

    if a.human:
        human_color = chess.WHITE if a.human == "white" else chess.BLACK
        play_game(net, human_color, a.depth, a.top_k)
    elif a.self_play:
        self_play(net, a.depth, a.top_k, games=a.games)
    else:
        print("Specify --human white|black or --self-play")


if __name__ == "__main__":
    main()
