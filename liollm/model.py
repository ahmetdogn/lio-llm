# Das eigentliche Modell: ein Decoder-only Transformer wie GPT.
#
# Ich habe nicht genau GPT-2 nachgebaut, sondern ein paar Sachen aus neueren Modellen
# (LLaMA) uebernommen, weil ich die verstehen wollte:
#   - RMSNorm statt LayerNorm
#   - RoPE (rotary embeddings) statt gelernter Positions-Embeddings
#   - SwiGLU als Feed-Forward Netz
#
# Quellen: "Attention is all you need", nanoGPT von Karpathy, LLaMA Paper

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 1024
    block_size: int = 256  # max. Kontextlaenge
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    rope_theta: float = 10000.0

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    # wie LayerNorm, nur ohne Mittelwert abziehen -> einfacher und laut LLaMA genauso gut
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ---- RoPE ----
# Die Positionen werden nicht addiert, sondern q und k werden je nach Position
# "gedreht". Dadurch haengt q*k nur noch vom Abstand der Tokens ab.
# (Vorstellung: jedes Paar von Werten ist ein 2D-Vektor, der um einen Winkel gedreht wird)


def precompute_rope(head_dim, max_len, theta):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(max_len).float()
    angles = torch.outer(pos, freqs)  # (max_len, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    # x hat die Form (B, n_head, T, head_dim)
    # immer zwei benachbarte Werte bilden ein Paar, das gedreht wird (2D Rotation)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)  # q, k, v auf einmal
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, kv_cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # KV-Cache: beim Generieren muss man k und v von alten Tokens nicht jedes
        # mal neu berechnen, sondern speichert sie einfach
        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"] = k
            kv_cache["v"] = v

        # Wenn nur ein neues Token reinkommt (T=1), darf es alles im Cache sehen,
        # deshalb braucht man dann keine Maske
        # (scaled_dot_product_attention macht softmax(q*k^T / sqrt(d)) * v)
        drop = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1), dropout_p=drop)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Heads wieder zusammenfuegen
        return self.resid_drop(self.proj(y))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 8/3 * n_embd damit es ungefaehr gleich viele Parameter hat wie das normale MLP
        # mit 4 * n_embd (weil es hier 3 statt 2 Matrizen gibt)
        hidden = int(8 * cfg.n_embd / 3)
        hidden = 64 * math.ceil(hidden / 64)
        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    # ein Transformer Block (Pre-Norm, also Norm vor Attention/MLP)
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, kv_cache)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # weight tying: Embedding und Output-Layer teilen sich die Gewichte (wie bei GPT-2)
        self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope(cfg.n_embd // cfg.n_head, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # aus dem GPT-2 Paper: die Projektionen am Ende vom Residual-Pfad kleiner initialisieren
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, kv_caches=None, start_pos=0):
        T = idx.shape[1]
        assert start_pos + T <= self.cfg.block_size, "Sequenz ist laenger als block_size"
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]

        x = self.drop(self.tok_emb(idx))
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, cos, sin, cache)
        x = self.norm_f(x)

        if targets is None:
            # beim Generieren braucht man nur die letzte Position
            return self.lm_head(x[:, -1:, :]), None

        logits = self.lm_head(x)
        # reshape statt view, view hat hier einen Fehler geworfen (nicht contiguous)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def configure_optimizer(self, lr, weight_decay):
        # weight decay nur auf Matrizen, nicht auf Norm-Gewichte (wie in nanoGPT)
        decay = [p for p in self.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        self.eval()
        idx = idx[:, -self.cfg.block_size:]
        caches = [{} for _ in self.blocks]
        logits, _ = self(idx, kv_caches=caches)  # erst den ganzen Prompt durchschicken
        pos = idx.shape[1]

        for _ in range(max_new_tokens):
            logits = logits[:, -1, :]

            if temperature == 0:
                # greedy, nimmt immer das wahrscheinlichste Token
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    # alles ausser den top_k besten Tokens auf -inf setzen
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                if top_p is not None:
                    # nucleus sampling: nur die Tokens behalten, die zusammen top_p
                    # Wahrscheinlichkeit haben
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    probs = sorted_logits.softmax(-1)
                    cum = probs.cumsum(-1)
                    sorted_logits[cum - probs > top_p] = -float("inf")
                    logits = torch.full_like(logits, -float("inf"))
                    logits = logits.scatter(1, sorted_idx, sorted_logits)
                probs = logits.softmax(-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)

            if pos >= self.cfg.block_size:
                # Kontext ist voll -> nur die letzten block_size Tokens nehmen und den
                # Cache neu aufbauen (nicht super effizient, funktioniert aber)
                window = idx[:, -self.cfg.block_size:]
                caches = [{} for _ in self.blocks]
                logits, _ = self(window, kv_caches=caches)
                pos = window.shape[1]
            else:
                logits, _ = self(next_id, kv_caches=caches, start_pos=pos)
                pos += 1

        return idx
