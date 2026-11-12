import numpy as np

N_OUT = 4096

def make_mask(L):
    """L: (B, maxlen) اندیس حرکات قانونی با پد -1  ->  ماسک بولی (B, 4096)"""
    mask = np.zeros((L.shape[0], N_OUT), dtype=bool)
    r, c = np.nonzero(L >= 0)
    mask[r, L[r, c]] = True
    return mask

class MLP:
    def __init__(self, n_in, n_hid, seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": (rng.normal(0, 1, (n_in, n_hid)) * np.sqrt(2.0 / n_in)).astype(dtype),
            "b1": np.zeros(n_hid, dtype=dtype),
            "W2": (rng.normal(0, 1, (n_hid, N_OUT)) * np.sqrt(1.0 / n_hid)).astype(dtype),
            "b2": np.zeros(N_OUT, dtype=dtype),
        }

    def forward(self, X, mask, drop=0.0, rng=None):
        P = self.params
        z1 = X @ P["W1"] + P["b1"]
        h = np.maximum(z1, 0)
        keep = None
        if drop > 0:
            keep = ((rng.random(h.shape) >= drop) / (1.0 - drop)).astype(h.dtype)
            h = h * keep
        logits = h @ P["W2"] + P["b2"]
        logits = np.where(mask, logits, -np.inf)          # فقط حرکات قانونی
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        return p, (X, z1, h, keep)

    def predict(self, X, mask):
        return self.forward(X, mask)[0]

    def loss_and_grads(self, X, y, mask, drop=0.0, rng=None, wd=0.0):
        P = self.params
        p, (X, z1, h, keep) = self.forward(X, mask, drop, rng)
        B = X.shape[0]
        ar = np.arange(B)
        loss = -np.mean(np.log(p[ar, y] + 1e-12))

        d = p.copy()
        d[ar, y] -= 1.0
        d /= B                                   # dLoss/dlogits (برای غیرقانونی‌ها صفر)
        g = {"W2": h.T @ d, "b2": d.sum(axis=0)}
        dh = d @ P["W2"].T
        if keep is not None:
            dh = dh * keep
        dz = dh * (z1 > 0)
        g["W1"] = X.T @ dz
        g["b1"] = dz.sum(axis=0)
        if wd > 0:
            g["W1"] = g["W1"] + wd * P["W1"]
            g["W2"] = g["W2"] + wd * P["W2"]
        return loss, g

class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)
