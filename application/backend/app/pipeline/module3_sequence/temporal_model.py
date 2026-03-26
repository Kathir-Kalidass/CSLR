"""
Temporal Model
BiLSTM or Transformer for sequence modeling
"""

from typing import Optional
import torch
import torch.nn as nn


class TemporalModel(nn.Module):
    """
    Temporal Sequence Model
    BiLSTM or Transformer encoder
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        vocab_size: int = 1000,
        model_type: str = "bilstm",  # bilstm or transformer
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        
        if model_type == "bilstm":
            self.encoder = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            encoder_output_dim = hidden_dim * 2  # Bidirectional
        
        elif model_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )
            encoder_output_dim = input_dim
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Output projection (for CTC)
        self.classifier = nn.Linear(encoder_output_dim, vocab_size + 1)  # +1 for CTC blank
        
        self.vocab_size = vocab_size
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (B, T, input_dim) - Feature sequences
            mask: (B, T) - Attention mask
        
        Returns:
            (B, T, vocab_size+1) - Logits for CTC
        """
        encoded: torch.Tensor
        
        if self.model_type == "bilstm":
            # BiLSTM encoding
            encoded, _ = self.encoder(x)  # (B, T, hidden_dim*2)
        
        elif self.model_type == "transformer":
            # Transformer encoding
            if mask is not None:
                # Create attention mask
                attn_mask = (mask == 0).unsqueeze(1).repeat(1, mask.size(1), 1)
                encoded = self.encoder(x, src_key_padding_mask=attn_mask)
            else:
                encoded = self.encoder(x)
        else:
            encoded = x
        
        # Project to vocabulary
        logits = self.classifier(encoded)  # (B, T, vocab_size+1)
        
        return logits


class SequenceEncoder(nn.Module):
    """
    Full sequence encoding pipeline
    Combines feature extraction and temporal modeling
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        vocab_size: int = 1000
    ):
        super().__init__()
        
        self.temporal_model = TemporalModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            vocab_size=vocab_size
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode feature sequence
        
        Args:
            features: (B, T, input_dim)
        
        Returns:
            (B, T, vocab_size+1) - CTC logits
        """
        return self.temporal_model(features)
