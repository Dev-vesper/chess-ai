# play — بازی و تحلیل با MLP prior

این پکیج فقط چرخه‌ی بازی/تحلیل تعاملی را نگه می‌دارد. بارگذاری مدل در `policy/` و جست‌وجو در `engine/` است.

## فایل‌ها
- `play.py` — CLI (self-play / human / fen)

## اجرا

```bash
# تحلیل موقعیت
python -m play.play --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --depth 3

# self-play
python -m play.play --self-play --depth 3 --games 1

# بازی با انسان
python -m play.play --human white --depth 3
python -m play.play --human black --depth 3
```
