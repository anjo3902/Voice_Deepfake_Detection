import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .feature_extractor import FeatureExtractor
except ImportError:
    from feature_extractor import FeatureExtractor

class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer for spectro-temporal modeling"""
    
    def __init__(self, in_dim, out_dim, num_heads=4):
        super(GraphAttentionLayer, self).__init__()
        
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        # Multi-head attention components
        self.q_linear = nn.Linear(in_dim, out_dim)
        self.k_linear = nn.Linear(in_dim, out_dim)
        self.v_linear = nn.Linear(in_dim, out_dim)
        
        self.out_linear = nn.Linear(out_dim, out_dim)
        
        self.layer_norm = nn.LayerNorm(out_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Multi-head attention
        Q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.out_dim)
        
        out = self.out_linear(out)
        out = self.layer_norm(out + x if x.size(-1) == self.out_dim else out)
        
        return out


class HeterogeneousStackingGAL(nn.Module):
    """Heterogeneous Stacking Graph Attention Layer"""
    
    def __init__(self, in_dim, out_dim, num_heads=4):
        super(HeterogeneousStackingGAL, self).__init__()
        
        # Spectral graph attention
        self.spectral_gat = GraphAttentionLayer(in_dim, out_dim, num_heads)
        
        # Temporal graph attention
        self.temporal_gat = GraphAttentionLayer(in_dim, out_dim, num_heads)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(0.2)  # Reduced from 0.3 to allow more learning
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # Spectral modeling
        spectral_out = self.spectral_gat(x)
        
        # Temporal modeling
        temporal_out = self.temporal_gat(x)
        
        # Combine spectral and temporal information
        combined = torch.cat([spectral_out, temporal_out], dim=-1)
        out = self.fusion(combined)
        
        return out


class AASIST(nn.Module):
    """
    Audio Anti-Spoofing using Integrated Spectro-Temporal graph attention networks
    Optimized for RTX 2050 4GB GPU
    """
    
    def __init__(self, sample_rate=16000, num_classes=2):
        super(AASIST, self).__init__()
        
        # Feature extraction
        self.feature_extractor = FeatureExtractor(sample_rate=sample_rate)
        
        # Reshape for graph attention
        self.input_projection = nn.Linear(512, 256)
        
        # Stacked graph attention layers
        self.hs_gal1 = HeterogeneousStackingGAL(256, 256, num_heads=4)
        self.hs_gal2 = HeterogeneousStackingGAL(256, 256, num_heads=4)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Reduced from 0.5 to allow more learning
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, 1, time)
        
        # Extract features
        features = self.feature_extractor(x)  # (batch, 512)
        
        # Project to lower dimension
        features = self.input_projection(features)  # (batch, 256)
        features = features.unsqueeze(1)  # (batch, 1, 256)
        
        # Graph attention processing
        x = self.hs_gal1(features)
        x = self.hs_gal2(x)
        
        # Global pooling
        x = torch.mean(x, dim=1)  # (batch, 256)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class LightweightDetector(nn.Module):
    """
    Lightweight alternative for faster inference
    Based on simplified AASIST architecture
    """
    
    def __init__(self, sample_rate=16000, num_classes=2):
        super(LightweightDetector, self).__init__()
        
        self.feature_extractor = FeatureExtractor(sample_rate=sample_rate)
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits


def get_model(model_type='aasist', sample_rate=16000, num_classes=2):
    """Factory function to get model"""
    
    if model_type == 'aasist':
        return AASIST(sample_rate=sample_rate, num_classes=num_classes)
    elif model_type == 'lightweight':
        return LightweightDetector(sample_rate=sample_rate, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")