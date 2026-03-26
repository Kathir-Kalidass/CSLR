from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from app.data.isign_dataset import build_dataloaders
from app.utils.grammar_correction import GrammarCorrector
from train_isign import (
    ISignCSLRModel,
    _confidence_from_log_probs,
    decode_sequences,
    ids_to_tokens,
    load_checkpoint,
)


def resolve_backend_relative(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return BACKEND_DIR / path


def printable_tokens(tokens: list[str]) -> str:
    return " ".join(tokens) if tokens else "(empty prediction)"


def make_record(
    shown_idx: int,
    sample_name: str,
    confidence: float,
    reference: list[str],
    predicted: list[str],
    reference_sentence: str,
    predicted_sentence: str,
) -> dict[str, Any]:
    return {
        "shown_idx": shown_idx,
        "sample_name": sample_name,
        "confidence": confidence,
        "reference": reference,
        "predicted": predicted,
        "reference_sentence": reference_sentence,
        "predicted_sentence": predicted_sentence,
    }


def print_record(record: dict[str, Any]) -> None:
    print(f"sample_{record['shown_idx']}")
    print(f"  name: {record['sample_name']}")
    print(f"  confidence: {record['confidence']:.4f}")
    print(f"  reference_gloss: {printable_tokens(record['reference'])}")
    print(f"  predicted_gloss: {printable_tokens(record['predicted'])}")
    print(f"  reference_sentence: {record['reference_sentence']}")
    print(
        "  predicted_sentence: "
        f"{record['predicted_sentence'] or '(empty prediction)'}"
    )
    print("-" * 80)


def main() -> None:
    ckpt_path = BACKEND_DIR / "checkpoints/isign_pose_only_npy/best.pt"
    config_path = BACKEND_DIR / "checkpoints/isign_pose_only_npy/train_config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    data_dir = resolve_backend_relative(cfg["data_dir"])
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    _, _, test_loader, vocab = build_dataloaders(
        data_dir=str(data_dir),
        batch_size=1,
        num_frames=int(cfg["num_frames"]),
        frame_size=tuple(cfg["frame_size"]),
        num_workers=0,
        use_poses=True,
        use_frames=not bool(cfg.get("pose_only", False)),
        max_samples=0,
        pin_memory=False,
        persistent_workers=False,
        preload_n=0,
        pose_backend=str(cfg.get("pose_backend", "auto")),
        pose_lmdb_path=cfg.get("pose_lmdb_path"),
        pose_lmdb_readahead=bool(cfg.get("pose_lmdb_readahead", False)),
    )

    first_batch = next(iter(test_loader))
    pose_input_dim = int(first_batch["pose"].shape[-1])
    use_rgb = not bool(cfg.get("pose_only", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ISignCSLRModel(
        vocab_size=len(vocab),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        pose_input_dim=pose_input_dim,
        dropout=float(cfg.get("dropout", 0.2)),
        pretrained_cnn=not bool(cfg.get("no_pretrained", False)),
        pose_fusion_weight=float(cfg.get("pose_fusion_weight", 0.7)),
        attention_heads=int(cfg.get("attention_heads", 4)),
        freeze_rgb_stages=int(cfg.get("freeze_rgb_stages", 4)),
        use_rgb=use_rgb,
    ).to(device)
    checkpoint = load_checkpoint(ckpt_path, model)
    model.eval()

    grammar = GrammarCorrector()

    print(f"checkpoint={ckpt_path}")
    print(f"epoch={checkpoint.get('epoch')} best_wer={checkpoint.get('best_wer')}")
    print(f"device={device.type}")
    print(f"data_dir={data_dir}")
    print(f"vocab_size={len(vocab)} pose_input_dim={pose_input_dim}")
    print("-" * 80)

    shown = 0
    non_empty = 0
    max_to_show = 5
    max_to_scan = 200
    fallback_records: list[dict[str, Any]] = []
    chosen_records: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader, start=1):
            if batch_idx > max_to_scan or len(chosen_records) >= max_to_show:
                break

            pose = batch["pose"].to(device)
            rgb = batch["rgb"].to(device) if use_rgb else None
            log_probs = model(rgb, pose)

            predicted = decode_sequences(log_probs, blank_id=0, vocab=vocab)[0]
            confidence = float(_confidence_from_log_probs(log_probs)[0].item())

            label_len = int(batch["label_lengths"][0].item())
            ref_ids = batch["labels"][0, :label_len].tolist()
            reference = ids_to_tokens(ref_ids, vocab, blank_id=0)

            sample_name = batch["names"][0]
            reference_sentence = batch["sentences"][0]
            predicted_sentence = grammar.gloss_to_sentence(predicted)

            record = make_record(
                shown_idx=shown + 1,
                sample_name=sample_name,
                confidence=confidence,
                reference=reference,
                predicted=predicted,
                reference_sentence=reference_sentence,
                predicted_sentence=predicted_sentence,
            )

            if len(fallback_records) < max_to_show:
                fallback_records.append(record)

            if predicted:
                non_empty += 1
                chosen_records.append(record)

            shown += 1

    output_records = chosen_records if chosen_records else fallback_records
    for idx, record in enumerate(output_records, start=1):
        record["shown_idx"] = idx
        print_record(record)

    print(
        f"shown={len(output_records)} scanned={min(max_to_scan, batch_idx)} "
        f"non_empty_predictions={non_empty}"
    )


if __name__ == "__main__":
    main()
