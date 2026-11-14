# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A framework-free chess move-prediction engine: a NumPy MLP trained on PGN games to predict the next move, plus an alpha-beta search that uses the MLP only as a root prior. There is no PyTorch, no packaged install, and no test suite.

Environment: Python 3.13 in `.venv`, dependencies `numpy` and `python-chess` only. There is no `requirements.txt`; use `.venv/bin/python` (or activate `.venv`).

All commands are module entry points run from the repo root. `dataset.npz` and `model.npz` are resolved relative to the CWD, so do not run from a subdirectory.

## Commands

```bash
# Build dataset from PGNs (games/ contains 479 files)
python -m dataset.build_dataset games --name a_ve3per --out dataset.npz

# Train (writes model.npz; only the best-val-loss epoch is saved)
python -m training.train --hidden 128 --epochs 40 --drop 0.5 --wd 1e-3 --lr 1e-3

# Numerical gradient check (the closest thing to a test suite)
python -m training.gradcheck

# Play / analyze
python -m play.play --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --depth 3
python -m play.play --self-play --depth 3 --games 1
python -m play.play --human white --depth 3

# Diagnostics
python -m dataset.inspect_pgn games       # PGN metadata/username counts
python -m policy.diagnose_prior           # model prior calibration from start position
python -m training.check_denormal         # denormal counts (--model defaults to model.npz)
```

## Package layout

Dependency direction (no cycles):

```
play ──► engine ──► policy.inference ──► policy.chessnet ──► policy.encoding
  └────────────────────────────────────────────────────────────►┘
dataset, training ──► policy.encoding / policy.chessnet
```

- `policy/` — learned move-policy domain. `encoding.py` is the single owner of the feature/move-space contract, model view/mirroring, and masking. `chessnet.py` is a pure-NumPy `MLP` + `Adam`. `inference.py` loads `.npz` parameters and converts a board into a move→probability dict.
- `dataset/` — PGN parsing and `dataset.npz` construction; `pgn.py` owns file enumeration.
- `training/` — training loop, validation split, and numerics diagnostics (gradient check, denormal check).
- `engine/` — handcrafted evaluation and alpha-beta search; no game-loop or CLI concerns.
- `play/` — interactive CLI/game loop only.

There are no root-level `.py` scripts and no `sys.path` manipulation; add new behavior to the owning package and run it with `python -m`.

## Data / model contracts (must stay in sync)

- **Feature vector, 805 dims** — owned by `policy/encoding.py::encode()`: 768 piece-square bits, 4 castling, 1 in-check, 5 friendly hanging types, 5 enemy hanging types, 11 material-balance buckets, 5 my center control, 5 opponent center control, 1 en-passant.
- **Action space, 4096 = `from_square * 64 + to_square`** — promotion piece is not part of the index, so promotions collapse onto one index. `move_index` / `move_from_index` / `legal_indices` in `encoding.py` are the only index helpers.
- **Side-to-move perspective** — `view_for_turn(board)` returns the board from the mover's view (Black positions are mirrored) plus a `mirrored` flag; `map_move(move, mirrored)` converts moves either way and preserves promotion. Use these instead of ad-hoc `board.mirror()` logic.
- **Dataset columns**: `X` (uint8 features), `Y` (int16 move index), `G` (game id), `P` (fullmove number), `L` (int16 legal indices padded with -1), `POS_KEY` (uint64 Polyglot Zobrist hash of the model view).
- **Validation split by position hash** — `training/train.py` assigns `POS_KEY % 100 < val_ratio*100` to validation, removes every validation key from training, and prints a leakage count that must stay 0.
- **Model file** is a bare parameter dict (`W1`, `b1`, `W2`, `b2`). `policy.inference.load_model()` wraps it with `MLP.from_params()` (no re-init, no dtype conversion) and asserts `W1.shape[0] == N_FEATURES`.

## Engine semantics

- `engine/evaluate.py` — material + mid/endgame PSTs, mobility (pseudo-legal counts), hanging penalties, pawn-shield king safety. `eval_negamax()` returns from the side-to-move perspective; `full=False` is the cheap quiescence variant.
- `engine/terminal.py` — `MATE` and draw predicates. The asymmetry is intentional: `is_search_draw` excludes automatic 75-move/fivefold, while `is_eval_draw` includes them; do not unify without a deliberate behavior change.
- `engine/search.py` — alpha-beta with capture/promotion quiescence and repetition tracking via a `seen` dict keyed by `board._transposition_key()` (private API, no public equivalent). The MLP affects only the root: legal moves are sorted by prior, truncated to `top_k`, then re-ranked by `score + λ·log(prior)` with `λ = clamp(tau·std(root scores), 1, 500)`. Mate scores bypass the prior.
- Interior nodes are pure heuristic search; only root move ordering uses the network. Do not describe or "fix" this as if an interior policy/value model were intended.

## Known remaining constraints

- No dependency manifest or test runner; `training/gradcheck.py` is the correctness gate.
- Artifact paths are CWD-relative by design.
- `.gitignore`'s first line (`# games/`) is a comment, so all 479 PGNs are tracked as the data source; `dataset.npz` and `model.npz` are ignored.
