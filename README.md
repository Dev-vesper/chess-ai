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

All commands must be run from the repository root. The model is a move-prior for the search, not a standalone engine.
