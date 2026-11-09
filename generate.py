# Text generieren mit einem trainierten Modell
#
#   python generate.py --prompt "ROMEO:"

import argparse

import torch

from liollm.model import GPT, ModelConfig
from liollm.tokenizer import BPETokenizer


def load_model(ckpt_path="checkpoints/ckpt.pt", tok_path="data/tokenizer.json"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = GPT(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer.load(tok_path)
    return model, tok


def generate_text(model, tok, prompt, max_new_tokens=200, temperature=0.8, top_p=0.9):
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_p=top_p)
    return tok.decode(out[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, tok = load_model()
    print(generate_text(model, tok, args.prompt, args.max_new_tokens, args.temperature, args.top_p))
