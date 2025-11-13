import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SincConv(nn.Module):
    """Sinc-based convolution for raw waveform processing"""
    
    def __init__(self, in_channels, out_channels, kernel_size, sample_rate=16000):
        super(SincConv, self).__init__()
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        
        # Initialize filterbank
        low_hz = 30
        high_hz = sample_rate / 2 - (low_hz)
        
        # Mel scale initialization
        mel = np.linspace(0, self._hz_to_mel(high_hz), self.out_channels + 1)
        hz = self._mel_to_hz(mel)
        
        # Filter bank parameters (learnable)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        
        # Hamming window (register as buffer so it moves with the model)
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        window = 0.54 - 0.46 * torch.cos(2 * np.pi * n_lin / self.kernel_size)
        self.register_buffer('window_', window)
        
        # Time grid for sinc filter (register as buffer)
        n = (self.kernel_size - 1) / 2.0
        time_grid = torch.arange(-n, 0).view(1, -1) / self.sample_rate
        self.register_buffer('time_grid_', time_grid)
        
    def _hz_to_mel(self, hz):
        return 2595 * np.log10(1 + hz / 700)
    
    def _mel_to_hz(self, mel):
        return 700 * (10 ** (mel / 2595) - 1)
    
    def forward(self, x):
        # Compute filters
        low = self.low_hz_.abs()
        high = torch.clamp(low + self.band_hz_.abs(), 0, self.sample_rate / 2)
        band = (high - low)[:, 0]
        
        f_times_t_low = torch.matmul(low, self.time_grid_)
        f_times_t_high = torch.matmul(high, self.time_grid_)
        
        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / 
                         (f_times_t_high - f_times_t_low)) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])
        
        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        
        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2, dilation=1, groups=1)


class ResBlock(nn.Module):
    """Residual block with pre-activation"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                                     stride=stride, bias=False)
    
    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class FeatureExtractor(nn.Module):
    """Enhanced feature extraction with Sinc filters and residual blocks"""
    
    def __init__(self, sample_rate=16000):
        super(FeatureExtractor, self).__init__()
        
        # Sinc-based first layer
        self.sinc_conv = SincConv(1, 128, kernel_size=251, sample_rate=sample_rate)
        
        # Residual blocks for feature learning
        self.layer1 = self._make_layer(128, 128, 2, stride=2)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_layer(256, 512, 2, stride=2)
        
        # Attention pooling
        self.attention = nn.Sequential(
            nn.Conv1d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, 512, kernel_size=1),
            nn.Sigmoid()
        )
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape: (batch, 1, time)
        
        # Sinc convolution
        x = torch.abs(self.sinc_conv(x))
        x = F.max_pool1d(x, kernel_size=3, stride=3)
        
        # Residual feature extraction
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Attention-weighted pooling
        att_weights = self.attention(x)
        x = x * att_weights
        x = torch.mean(x, dim=2)
        
        return x