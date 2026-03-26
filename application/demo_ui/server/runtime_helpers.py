from __future__ import annotations

from typing import Any, Optional

from .deps import torch


def sub_state(state: dict[str, Any], prefix: str) -> dict[str, Any]:
    prefix_dot = f"{prefix}."
    return {k[len(prefix_dot) :]: v for k, v in state.items() if k.startswith(prefix_dot)}


def infer_temporal_layers(state: dict[str, Any]) -> int:
    layers: set[int] = set()
    for key in state:
        if not key.startswith("temporal.encoder.weight_ih_l"):
            continue
        tail = key.split("temporal.encoder.weight_ih_l", maxsplit=1)[1]
        idx_raw = tail.split("_", maxsplit=1)[0]
        if idx_raw.isdigit():
            layers.add(int(idx_raw))
    return (max(layers) + 1) if layers else 2


def build_id_to_token(checkpoint: dict[str, Any], vocab_size: int) -> dict[int, str]:
    raw = checkpoint.get("id_to_token")
    out: dict[int, str] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                token_id = int(key)
            except Exception:
                continue
            out[token_id] = str(value)
    if not out:
        out = {idx: f"TOKEN_{idx}" for idx in range(1, vocab_size + 1)}
    return out


def ctc_decode(logits: Any, id_to_token: dict[int, str]) -> tuple[list[str], float]:
    probs = torch.softmax(logits, dim=-1)
    confidence = float(torch.max(probs, dim=-1).values.mean().item())
    pred = torch.argmax(logits, dim=-1).cpu().numpy().tolist()

    tokens: list[int] = []
    prev: Optional[int] = None
    for idx in pred:
        if idx != 0 and idx != prev:
            tokens.append(int(idx))
        prev = int(idx)

    decoded = [id_to_token.get(idx, f"TOKEN_{idx}") for idx in tokens]
    return decoded, confidence


def gloss_to_sentence(gloss_tokens: list[str]) -> str:
    if not gloss_tokens:
        return ""
    text = " ".join(gloss_tokens)
    if text.startswith("ME GO "):
        return f"I am going to {text[6:].lower()}."
    if text.startswith("ME NEED "):
        return f"I need {text[8:].lower()}."
    if text in {"HELLO", "HI"}:
        return "Hello."
    if text in {"THANK-YOU", "THANKS"}:
        return "Thank you."
    return f"{text.capitalize()}."
