<div align="center">

# chess-ai

A framework-free chess move-prediction engine: a NumPy MLP trained on PGN games, with alpha-beta search using the network as a root prior.

[![Python 3.13](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy only](https://img.shields.io/badge/NumPy-only-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![python-chess](https://img.shields.io/badge/python--chess-1.x-4B8BBE)](https://python-chess.readthedocs.io/)

</div>

## Purpose

This framework is meant solely for training on your own personal games. It learns your playing style from your PGN history, and the alpha-beta engine then plays using that learned style as its prior.

## How it works

- **Features:** 805 board features (piece squares, castling, checks, hanging pieces, material, center control, en passant).
- **Model:** a NumPy MLP (`805 → 128 → 4096`) predicting a softmax over `from_square * 64 + to_square` moves.
- **Search:** alpha-beta with handcrafted evaluation. The model only orders and filters root moves (`top_k`) and adds a `λ·log(prior)` bonus; interior nodes are pure search.

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install numpy python-chess
```

## Build the dataset

`games/` contains 479 PGN files. `--name` is your username as written in the PGN headers:

```bash
python -m dataset.build_dataset games --name a_ve3per --out dataset.npz
```

This writes `dataset.npz` with features (`X`), move targets (`Y`), legal-move lists (`L`), and Zobrist position keys (`POS_KEY`).

## Train the model

```bash
python -m training.train --hidden 128 --epochs 40 --drop 0.5 --wd 1e-3 --lr 1e-3
```

Training saves `model.npz` from the best validation-loss epoch. Validation is split by position hash with zero leakage. Use `--out` to write elsewhere, and run the gradient check with:

```bash
python -m training.gradcheck
```

## Play

```bash
# Analyze a position
python -m play.play --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --depth 3

# Self-play
python -m play.play --self-play --depth 3 --games 1

# Play against the engine
python -m play.play --human white --depth 3
```

## CLI options

### `play.play`

| Flag | Default | Description |
| --- | --- | --- |
| `--fen FEN` | — | Analyze one position instead of playing a game. |
| `--depth N` | `3` | Alpha-beta search depth. |
| `--top-k N` | `8` | Search only the top-N root moves by model prior. `0` disables truncation. |
| `--tau X` | `1.0` | Prior strength: `λ = clamp(tau × std(root scores), 1, 500)`. Larger = model has more influence; smaller = search dominates. |
| `--model PATH` | `model.npz` | Trained parameter file. |
| `--human {white,black}` | — | Play interactively as the chosen side. |
| `--self-play` | off | Let the engine play both sides. |
| `--games N` | `1` | Number of self-play games. |

### `dataset.build_dataset`

| Argument | Default | Description |
| --- | --- | --- |
| `pgn` | required | PGN file or directory of PGN files. |
| `--name NAME [NAME ...]` | required | Username(s) whose moves become training targets. |
| `--out PATH` | `dataset.npz` | Output dataset file. |

### `training.train`

| Flag | Default | Description |
| --- | --- | --- |
| `--hidden N` | `128` | Hidden-layer width. |
| `--epochs N` | `40` | Maximum training epochs. |
| `--batch N` | `64` | Batch size. |
| `--lr X` | `1e-3` | Adam learning rate. |
| `--drop X` | `0.5` | Dropout rate. |
| `--wd X` | `1e-3` | Weight decay. |
| `--patience N` | `5` | Early stop after N epochs without validation improvement. |
| `--val_ratio X` | `0.15` | Fraction of positions (by Zobrist hash) held out for validation. |
| `--seed N` | `0` | Random seed. |
| `--flush N` | `1` | Flush tiny Adam moment values to zero (`1` on, `0` off). |
| `--out PATH` | `model.npz` | Output model file. |

`training.check_denormal` accepts `--model PATH` (default `model.npz`). `training.gradcheck` and `policy.diagnose_prior` take no flags, and `dataset.inspect_pgn` takes only a PGN path.

All commands must be run from the repository root. The model is a move-prior for the search, not a standalone engine.
