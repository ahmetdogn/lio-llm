# Das gleiche Modell wie in model.py, aber nur mit numpy.
#
# Wofuer? PyTorch ist zu gross fuer Vercel (dort gibt es ein Limit fuer die Groesse der
# Serverless Functions). Zum Generieren braucht man aber nur den Forward-Pass, und den kann
# man auch mit numpy machen. Die Gewichte werden vorher mit export_numpy.py exportiert.

import numpy as np


def rmsnorm(x, w, eps=1e-6):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def silu(x):
    return x / (1 + np.exp(-x))


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


class NumpyGPT:
    def __init__(self, path):
        data = np.load(path)
        # fp16 gespeichert (halbe Dateigroesse), gerechnet wird aber in fp32
        self.w = {k: data[k].astype(np.float32) for k in data.files if k != "config"}
        cfg = data["config"]
        self.vocab_size, self.block_size, self.n_layer, self.n_head, self.n_embd = (int(c) for c in cfg)
        self.head_dim = self.n_embd // self.n_head

        # RoPE Winkel vorberechnen (wie in model.py)
        freqs = 1.0 / (10000.0 ** (np.arange(0, self.head_dim, 2) / self.head_dim))
        angles = np.outer(np.arange(self.block_size), freqs)
        self.cos = np.cos(angles).astype(np.float32)
        self.sin = np.sin(angles).astype(np.float32)

    def rope(self, x, start):
        # x: (n_head, T, head_dim)
        T = x.shape[1]
        cos = self.cos[start:start + T]
        sin = self.sin[start:start + T]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out

    def forward(self, ids, cache, start):
        # ids: Liste von Token-IDs, cache: Liste mit (k, v) pro Layer
        # gibt die Logits fuer die letzte Position zurueck
        w = self.w
        T = len(ids)
        H, D = self.n_head, self.head_dim
        x = w["tok_emb.weight"][ids]  # (T, C)

        for i in range(self.n_layer):
            p = f"blocks.{i}."
            h = rmsnorm(x, w[p + "norm1.weight"])
            qkv = h @ w[p + "attn.qkv.weight"].T
            q, k, v = np.split(qkv, 3, axis=-1)
            # (T, C) -> (H, T, D)
            q = q.reshape(T, H, D).transpose(1, 0, 2)
            k = k.reshape(T, H, D).transpose(1, 0, 2)
            v = v.reshape(T, H, D).transpose(1, 0, 2)
            q = self.rope(q, start)
            k = self.rope(k, start)

            if cache[i] is not None:
                k = np.concatenate([cache[i][0], k], axis=1)
                v = np.concatenate([cache[i][1], v], axis=1)
            cache[i] = (k, v)

            att = q @ k.transpose(0, 2, 1) / np.sqrt(D)  # (H, T, T_gesamt)
            if T > 1:
                # kausale Maske
                mask = np.triu(np.ones((T, k.shape[1]), dtype=bool), k=k.shape[1] - T + 1)
                att[:, mask] = -np.inf
            y = softmax(att) @ v  # (H, T, D)
            y = y.transpose(1, 0, 2).reshape(T, H * D)
            x = x + y @ w[p + "attn.proj.weight"].T

            h = rmsnorm(x, w[p + "norm2.weight"])
            gate = silu(h @ w[p + "mlp.w_gate.weight"].T)
            up = h @ w[p + "mlp.w_up.weight"].T
            x = x + (gate * up) @ w[p + "mlp.w_down.weight"].T

        x = rmsnorm(x[-1], w["norm_f.weight"])
        return x @ w["tok_emb.weight"].T  # weight tying

    def generate(self, ids, max_new_tokens, temperature=0.8, top_p=0.9, rng=None):
        rng = rng or np.random.default_rng()
        ids = list(ids)[-self.block_size:]
        cache = [None] * self.n_layer
        logits = self.forward(ids, cache, 0)
        pos = len(ids)

        for _ in range(max_new_tokens):
            logits = logits / temperature
            # top-p sampling
            order = np.argsort(-logits)
            probs = softmax(logits[order])
            cum = np.cumsum(probs)
            keep = (cum - probs) <= top_p
            probs = probs * keep
            probs = probs / probs.sum()
            next_id = int(order[rng.choice(len(probs), p=probs)])
            ids.append(next_id)

            if pos >= self.block_size:
                # Kontext voll -> Fenster verschieben und Cache neu aufbauen
                window = ids[-self.block_size:]
                cache = [None] * self.n_layer
                logits = self.forward(window, cache, 0)
                pos = len(window)
            else:
                logits = self.forward([next_id], cache, pos)
                pos += 1
        return ids
