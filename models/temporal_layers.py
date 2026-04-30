import torch
import torch.nn as nn
import math

class SimpleTransformerTemporalModel(nn.Module):
    def __init__(self, 
                d_model, 
                nhead=8, 
                num_layers=2, 
                dim_feedforward=2048, 
                dropout=0.1, 
                pe_choice=None, 
                max_seq_len=1000, 
                temporal_pool='avg'):
        super(SimpleTransformerTemporalModel, self).__init__()
        self.d_model = d_model
        self.temporal_pool = temporal_pool
        self.pe_choice = pe_choice
        
        if pe_choice == "learnable":
            # Learnable positional embedding
            self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model))
        elif pe_choice == "sinusoidal":
            # Fixed sinusoidal positional embedding
            self.register_buffer('pos_embedding', self._create_sinusoidal_pe(max_seq_len, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def _create_sinusoidal_pe(self, max_len, d_model):
        """Create fixed sinusoidal positional embeddings"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        
        # Add positional embeddings
        if self.pe_choice != 'None':
            pos_emb = self.pos_embedding[:seq_len, :].unsqueeze(0)  # (1, seq_len, d_model)
            x = x + pos_emb
        
        # Apply transformer
        output = self.transformer(x)  # (batch_size, seq_len, d_model)
        
        # Temporal pooling: average or sum
        if self.temporal_pool == 'avg':
            output = output.mean(dim=1)  # (batch_size, d_model)
        elif self.temporal_pool == 'sum':
            output = output.sum(dim=1)  # (batch_size, d_model)
        elif self.temporal_pool == "last_layer":
            output = output[:, -1, :]
        # If 'none', return full sequence (batch_size, seq_len, d_model)
        
        return output

# Example usage:
# With average pooling
# model_avg = SimpleTransformerTemporalModel(d_model=512, temporal_pool='avg')

# With sum pooling
# model_sum = SimpleTransformerTemporalModel(d_model=512, temporal_pool='sum')

# Without pooling (return full sequence)
# model_none = SimpleTransformerTemporalModel(d_model=512, temporal_pool='none')

# features = torch.randn(2, 50, 512)  # (batch_size=2, seq_len=50, d_model=512)
# output = model_avg(features)  # (2, 512) with avg, (2, 512) with sum, (2, 50, 512) with none