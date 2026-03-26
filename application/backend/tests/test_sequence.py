"""
Test Sequence Modeling Module
"""

import pytest
import torch


def test_temporal_model_bilstm():
    """Test BiLSTM temporal model"""
    from app.pipeline.module3_sequence.temporal_model import TemporalModel
    
    model = TemporalModel(
        input_dim=512,
        hidden_dim=256,
        num_layers=2,
        vocab_size=100,
        model_type="bilstm"
    )
    
    assert model is not None


def test_temporal_model_forward():
    """Test temporal model forward pass"""
    from app.pipeline.module3_sequence.temporal_model import TemporalModel
    
    model = TemporalModel(
        input_dim=512,
        hidden_dim=256,
        num_layers=2,
        vocab_size=100,
        model_type="bilstm"
    )
    model.eval()
    
    # Input: (B, T, feature_dim)
    x = torch.randn(1, 16, 512)
    
    with torch.no_grad():
        logits = model(x)
    
    # Output: (B, T, vocab_size)
    assert logits.shape == (1, 16, 100)


def test_ctc_layer():
    """Test CTC layer"""
    from app.pipeline.module3_sequence.ctc_layer import CTCLayer
    
    ctc = CTCLayer(vocab_size=100)
    
    # Create dummy logits and targets
    logits = torch.randn(16, 1, 100)  # (T, B, vocab_size)
    targets = torch.tensor([[1, 2, 3]])  # (B, S)
    input_lengths = torch.tensor([16])
    target_lengths = torch.tensor([3])
    
    loss = ctc(logits, targets, input_lengths, target_lengths)
    
    assert loss is not None
    assert loss.item() >= 0


def test_ctc_decoder_greedy():
    """Test greedy CTC decoding"""
    from app.pipeline.module3_sequence.decoder import CTCDecoder
    
    vocab = ["<blank>", "HELLO", "WORLD", "THANKS"]
    decoder = CTCDecoder(labels=vocab, blank_id=0)
    
    # Create dummy logits
    logits = torch.randn(1, 20, 4)  # (B, T, vocab_size)
    
    decoded = decoder.greedy_decode(logits)
    
    assert isinstance(decoded, list)
    assert len(decoded) == 1  # Batch size 1


def test_ctc_decoder_beam_search():
    """Test beam search CTC decoding"""
    from app.pipeline.module3_sequence.decoder import CTCDecoder
    
    vocab = ["<blank>", "A", "B", "C"]
    decoder = CTCDecoder(labels=vocab, beam_width=5)
    
    # Create dummy logits
    logits = torch.randn(1, 10, 4)
    
    decoded = decoder.beam_search_decode(logits)
    
    assert isinstance(decoded, list)
    assert len(decoded) == 1


def test_ctc_decoder_lm_rescoring_prefers_language_model_sequence():
    """LM rescoring should be able to rerank final beam candidates."""
    from app.utils.ctc_decoder import CTCDecoder

    class FakeLanguageModel:
        def score(self, text: str, bos: bool = True, eos: bool = True) -> float:
            return 5.0 if text == "A B" else 0.0

    vocab = ["<blank>", "A", "B"]
    logits = torch.tensor(
        [
            [0.0, 4.5, 2.0],
            [0.0, 4.2, 3.9],
        ],
        dtype=torch.float32,
    )

    acoustic_only = CTCDecoder(labels=vocab, blank_idx=0, beam_width=3)
    lm_decoder = CTCDecoder(
        labels=vocab,
        blank_idx=0,
        beam_width=3,
        lm_weight=1.0,
        lm_candidates=6,
        language_model=FakeLanguageModel(),
    )

    acoustic = acoustic_only.beam_search_decode(logits)
    rescored = lm_decoder.beam_search_decode(logits)

    assert acoustic.gloss_tokens == ["A"]
    assert rescored.gloss_tokens == ["A", "B"]
    assert rescored.lm_score is not None


def test_confidence_scoring():
    """Test confidence scoring"""
    from app.pipeline.module3_sequence.confidence import ConfidenceScorer
    
    scorer = ConfidenceScorer()
    
    # Create dummy logits
    logits = torch.randn(1, 10, 50)
    predictions = [["HELLO", "WORLD"]]
    
    scores = scorer.compute_confidence(logits, predictions)
    
    assert isinstance(scores, list)
    assert len(scores) == 1
    assert 0.0 <= scores[0] <= 1.0
