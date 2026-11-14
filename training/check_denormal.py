import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.npz")
    a = ap.parse_args()

    m = np.load(a.model)
    tiny = np.finfo(np.float32).tiny   # کوچک‌ترین عدد نرمال float32 ≈ 1.18e-38
    for k in m.files:
        w = m[k]
        nz = w != 0
        den = (np.abs(w) < tiny) & nz
        print(f"{k}: denormal count = {den.sum()} / {w.size}")


if __name__ == "__main__":
    main()
