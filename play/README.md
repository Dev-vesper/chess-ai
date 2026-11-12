# play — Alpha-Beta با MLP prior

## فایل‌ها
- `model_io.py` — بارگذاری مدل + گرفتن policy (با mirror برای black)
- `evaluate.py` — material + PST + hanging + king safety
- `search.py` — alpha-beta + quiescence + model bonus در root
- `play.py` — CLI (self-play / human / fen)
- `diagnose_prior.py` — بررسی calibration مدل

## اجرا

```bash
# تحلیل موقعیت
python -m play.play --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --depth 3

# self-play
python -m play.play --self-play --depth 3 --games 1

# بازی با انسان
python -m play.play --human white --depth 3
python -m play.play --human black --depth 3
