import sys, glob, collections
import chess.pgn

def pgn_files(path):
    return sorted(glob.glob(path + "/*.pgn")) if path.endswith("/") or not path.endswith(".pgn") else [path]

def main(path):
    names = collections.Counter()
    variants = collections.Counter()
    timectl = collections.Counter()
    results = collections.Counter()
    n_games = n_errors = n_empty = 0
    for fp in pgn_files(path):
        with open(fp, encoding="utf-8", errors="replace") as f:
            while True:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                n_games += 1
                h = g.headers
                names[h.get("White", "?")] += 1
                names[h.get("Black", "?")] += 1
                variants[h.get("Variant", "Standard")] += 1
                timectl[h.get("TimeControl", "?")] += 1
                results[h.get("Result", "?")] += 1
                if g.errors:
                    n_errors += 1
                if not any(True for _ in g.mainline_moves()):
                    n_empty += 1
    print("games:", n_games, "| with parse errors:", n_errors, "| empty:", n_empty)
    print("\nTop names (your username should be near the top):")
    for k, v in names.most_common(8): print(f"  {v:4d}  {k}")
    print("\nVariants:", dict(variants))
    print("Results:", dict(results))
    print("Time controls (top 6):", timectl.most_common(6))

if __name__ == "__main__":
    main(sys.argv[1])
