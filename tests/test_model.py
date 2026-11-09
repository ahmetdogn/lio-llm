import math

import torch

from liollm import GPT, ModelConfig

# kleines Modell, damit die Tests schnell laufen
CFG = ModelConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


def test_forward():
    torch.manual_seed(0)
    model = GPT(CFG)
    x = torch.randint(0, 64, (3, 16))
    y = torch.randint(0, 64, (3, 16))
    logits, loss = model(x, y)
    assert logits.shape == (3, 16, 64)
    # am Anfang sollte der Loss ungefaehr ln(64) sein (alle Tokens gleich wahrscheinlich)
    assert abs(loss.item() - math.log(64)) < 0.5


def test_kausal():
    # das Modell darf nicht in die Zukunft schauen:
    # wenn man das letzte Token aendert, duerfen sich die anderen Vorhersagen nicht aendern
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    x = torch.randint(0, 64, (1, 10))
    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 1) % 64
    a, _ = model(x, x)
    b, _ = model(x2, x2)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


def test_kv_cache():
    # mit Cache muessen die gleichen Ergebnisse rauskommen wie ohne
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    x = torch.randint(0, 64, (2, 12))
    full, _ = model(x, x)

    caches = [{} for _ in model.blocks]
    model(x[:, :8], kv_caches=caches)
    for t in range(8, 12):
        out, _ = model(x[:, t:t + 1], kv_caches=caches, start_pos=t)
        assert torch.allclose(out[:, -1], full[:, t], atol=1e-4)


def test_generate():
    # greedy generieren mit Cache vs. ganz einfach ohne Cache
    # (40 Tokens > block_size, damit auch das Sliding Window getestet wird)
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    prompt = torch.randint(0, 64, (1, 5))
    out = model.generate(prompt, max_new_tokens=40, temperature=0)
    assert out.shape == (1, 45)

    ref = prompt
    for _ in range(40):
        logits, _ = model(ref[:, -CFG.block_size:])
        ref = torch.cat([ref, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(out, ref)


def test_overfitting():
    # ein Modell sollte einen einzelnen kleinen Batch auswendig lernen koennen,
    # sonst ist irgendwo ein Bug
    torch.manual_seed(0)
    model = GPT(CFG)
    opt = model.configure_optimizer(lr=3e-3, weight_decay=0.0)
    x = torch.randint(0, 64, (2, 16))
    for _ in range(150):
        _, loss = model(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
